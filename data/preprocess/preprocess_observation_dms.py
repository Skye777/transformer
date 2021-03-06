"""
DMS: direct multi-step, don't need to predict all predictors
"""
import os
import json
import random
import numpy as np
import tensorflow as tf
import netCDF4 as nc
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split
from progress.bar import PixelBar

from hparams import Hparams

hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()


# ---------- Helpers ----------
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# ---------- Prepare Data ----------
def parse_npz_and_nc_data():
    height = hp.height
    width = hp.width

    train_scope = range(0, 193)   # 1982.12-1997.12
    sst = np.load(f"{hp.observe_npz_dir}/sst.npz")['sst'][train_scope, :, :]
    uwind = np.load(f"{hp.observe_npz_dir}/uwind.npz")['uwind'][train_scope, :, :]
    vwind = np.load(f"{hp.observe_npz_dir}/vwind.npz")['vwind'][train_scope, :, :]
    sshg = np.load(f"{hp.observe_npz_dir}/sshg.npz")['sshg'][train_scope, :, :]
    thflx = np.load(f"{hp.observe_npz_dir}/thflx.npz")['thflx'][train_scope, :, :]
    sst[abs(sst) < 8e-17] = 0

    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    # scaler = Normalizer()
    sst = np.reshape(scaler.fit_transform(np.reshape(sst, (-1, height * width))), (-1, height, width))
    uwind = np.reshape(scaler.fit_transform(np.reshape(uwind, (-1, height * width))), (-1, height, width))
    vwind = np.reshape(scaler.fit_transform(np.reshape(vwind, (-1, height * width))), (-1, height, width))
    sshg = np.reshape(scaler.fit_transform(np.reshape(sshg, (-1, height * width))), (-1, height, width))
    thflx = np.reshape(scaler.fit_transform(np.reshape(thflx, (-1, height * width))), (-1, height, width))

    data = []
    target = []
    for i in range(sst.shape[0] - hp.in_seqlen + 1 - hp.lead_time - hp.out_seqlen):
        data.append({'sst': sst[i:i + hp.in_seqlen].astype(np.float32),
                     'uwind': uwind[i:i + hp.in_seqlen].astype(np.float32),
                     'vwind': vwind[i:i + hp.in_seqlen].astype(np.float32),
                     'sshg': sshg[i:i + hp.in_seqlen].astype(np.float32),
                     'thflx': thflx[i:i + hp.in_seqlen].astype(np.float32)})
        
        target_start = i + hp.in_seqlen - 1 + hp.lead_time
        target.append({'sst': sst[target_start:target_start + hp.out_seqlen].astype(np.float32)
                       })

    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=hp.train_eval_split,
                                                                        random_state=hp.random_seed)
    print(len(train_data), len(test_data), len(train_target), len(test_target))
    return train_data, test_data, train_target, test_target


# ---------- IO ----------
def write_records(data, filename):
    series = data[0]
    target = data[1]
    writer = tf.io.TFRecordWriter(f'{hp.observe_preprocess_out_dir}/{filename}')

    bar = PixelBar(r'Generating', max=len(data), suffix='%(percent)d%%')
    for s, t in zip(series, target):
        example = tf.train.Example(features=tf.train.Features(feature={
            'input_sst': _bytes_feature(s['sst'].tobytes()),
            'input_uwind': _bytes_feature(s['uwind'].tobytes()),
            'input_vwind': _bytes_feature(s['vwind'].tobytes()),
            'input_sshg': _bytes_feature(s['sshg'].tobytes()),
            'input_thflx': _bytes_feature(s['thflx'].tobytes()),
            'output_sst': _bytes_feature(t['sst'].tobytes())
        }))
        writer.write(example.SerializeToString())
        bar.next()
    writer.close()
    bar.finish()


# ---------- Go! ----------
if __name__ == "__main__":
    if not os.path.exists(hp.observe_preprocess_out_dir):
        print("Creating output directory {}...".format(hp.observe_preprocess_out_dir))
        os.makedirs(hp.observe_preprocess_out_dir)

    print("Parsing raw data...")
    train_data, test_data, train_target, test_target = parse_npz_and_nc_data()
    print("Writing TF Records to file...")
    write_records((train_data, train_target), "train.tfrecords")
    write_records((test_data, test_target), "test.tfrecords")

    print("Done!")
