import tensorflow as tf
import numpy as np

from hparams import Hparams

hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()


def parse_fn(example):
    height = hp.height
    width = hp.width
    
    features = {
        'input_sst': tf.io.FixedLenFeature([], tf.string), 'input_uwind': tf.io.FixedLenFeature([], tf.string),
        'input_vwind': tf.io.FixedLenFeature([], tf.string), 'input_sshg': tf.io.FixedLenFeature([], tf.string),
        'input_thflx': tf.io.FixedLenFeature([], tf.string),
        'output_sst': tf.io.FixedLenFeature([], tf.string), 'output_uwind': tf.io.FixedLenFeature([], tf.string),
        'output_vwind': tf.io.FixedLenFeature([], tf.string), 'output_sshg': tf.io.FixedLenFeature([], tf.string),
        'output_thflx': tf.io.FixedLenFeature([], tf.string)}

    parsed = tf.io.parse_single_example(serialized=example, features=features)
    # print("parsed:", parsed)

    inputs_list = []
    outputs_list = []
    for vrb in hp.input_variables:
        inputs_list.append(tf.reshape(tf.io.decode_raw(parsed['input_'+vrb], tf.float32), [hp.in_seqlen, height, width, 1]))
    for vrb in hp.output_variables:
        outputs_list.append(tf.reshape(tf.io.decode_raw(parsed['output_'+vrb], tf.float32), [hp.out_seqlen, height, width, 1])[:, :, :, :])
    # print("inputs_list:", inputs_list)
    # print("outputs_list:", outputs_list)
    # [time, h, w, predictor]
    inputs_list = tf.transpose(tf.squeeze(inputs_list), [1, 2, 3, 0])
    outputs_list = tf.transpose(tf.squeeze(outputs_list), [1, 2, 3, 0])
    decoder_inp = tf.concat((tf.expand_dims(inputs_list[-1], 0), outputs_list[:-1]), axis=0)

    x = inputs_list
    ys = (decoder_inp, outputs_list)

    # input_features = {}
    # output_features = {}
    # for i, vrb in enumerate(hp.input_variables):
    #     input_features[vrb] = inputs_list[i]
    # for i, vrb in enumerate(hp.output_variables):
    #     output_features[vrb] = outputs_list[i]
    # print("input_feature:", input_features)
    # print("output_feature:", output_features)

    return x, ys


def train_input_fn():

    train_filenames = [hp.observe_preprocess_out_dir + "/train.tfrecords"]
    train_dataset = tf.data.TFRecordDataset(train_filenames)
    train_dataset = train_dataset.map(parse_fn)
    train_dataset = train_dataset.shuffle(hp.random_seed).batch(hp.batch_size)

    test_filenames = [hp.observe_preprocess_out_dir + "/test.tfrecords"]
    test_dataset = tf.data.TFRecordDataset(test_filenames)
    test_dataset = test_dataset.map(parse_fn)
    test_dataset = test_dataset.batch(hp.batch_size)

    return train_dataset, test_dataset


if __name__ == '__main__':
    train, test = train_input_fn()
    print("train:", train)
    print("test:", test)
