"""
@author: Skye Cui
@file: test_DMS.py
@time: 2021/5/3 9:59
@description: 
"""
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from model import UConvlstm
from component.plot_helper import plot_helper
from hparams import Hparams

hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()


def preprocess_helper(h, w, one_sample=False):
    test_month_start = hp.test_month_start
    sst = np.load(f"{hp.observe_npz_dir}/sst.npz")['sst']
    if one_sample:
        scope = range(test_month_start, test_month_start+hp.in_seqlen+hp.out_seqlen)
    else:
        scope = range(test_month_start, len(sst))
    sst = sst[scope, :, :]
    uwind = np.load(f"{hp.observe_npz_dir}/uwind.npz")['uwind'][scope, :, :]
    vwind = np.load(f"{hp.observe_npz_dir}/vwind.npz")['vwind'][scope, :, :]
    sshg = np.load(f"{hp.observe_npz_dir}/sshg.npz")['sshg'][scope, :, :]
    thflx = np.load(f"{hp.observe_npz_dir}/thflx.npz")['thflx'][scope, :, :]

    sst_origin = np.load(f"{hp.observe_npz_dir}/sst.npz")['sst'][scope, :, :]

    sst[abs(sst) < 8e-17] = 0

    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    # scaler = Normalizer()
    sst = np.reshape(scaler.fit_transform(np.reshape(sst, (-1, h * w))), (1, -1, h, w))
    uwind = np.reshape(scaler.fit_transform(np.reshape(uwind, (-1, h * w))), (1, -1, h, w))
    vwind = np.reshape(scaler.fit_transform(np.reshape(vwind, (-1, h * w))), (1, -1, h, w))
    sshg = np.reshape(scaler.fit_transform(np.reshape(sshg, (-1, h * w))), (1, -1, h, w))
    thflx = np.reshape(scaler.fit_transform(np.reshape(thflx, (-1, h * w))), (1, -1, h, w))

    return sst, uwind, vwind, sshg, thflx, sst_origin


def nino_seq(ssta):
    # inputs: [sample, time, h, w]
    # outputs: [sample, time]
    nino = []
    for sample in range(len(ssta)):
        n_index = [np.mean(ssta[sample, i, 70:90, 140:240]) for i in range(len(ssta[sample]))]
        nino.append(n_index)
    return nino


def get_rmse(y_true, y_preds):
    return np.sqrt(mean_squared_error(y_pred=y_preds, y_true=y_true))


def pcc(y_true, y_preds):
    # inputs: [sample, time]
    accskill = 0
    cor = []
    y_true_mean = np.mean(y_true, axis=0)
    y_pred_mean = np.mean(y_preds, axis=0)

    for i in range(hp.out_seqlen):
        fenzi = np.sum((y_true[:, i] - y_true_mean[i]) * (y_preds[:, i] - y_pred_mean[i]))
        fenmu = np.sqrt(
            np.sum((y_true[:, i] - y_true_mean[i]) ** 2) * np.sum((y_preds[:, i] - y_pred_mean[i]) ** 2))
        cor_i = fenzi / fenmu
        cor.append(cor_i)
        accskill += cor_i

    return accskill, cor


def predict():
    h = hp.height
    w = hp.width

    sst, uwind, vwind, sshg, thflx, sst_origin = preprocess_helper(h, w)

    cli_mean = np.mean(sst_origin, axis=0)
    # cli_mean = cli_mean[140:180, 120:320]

    sst_scaler = MinMaxScaler()
    sst_scaler.fit_transform(np.reshape(sst_origin, (-1, h * w)))

    sst_true = []
    sst_preds = []
    ssta_true = []
    ssta_preds = []
    test_samples = []
    for m in range(sst.shape[0] - hp.in_seqlen + 1 - hp.lead_time - hp.out_seqlen):
        data = np.transpose(
            [sst[:, m:m + hp.in_seqlen, :, :], uwind[:, m:m + hp.in_seqlen, :, :], vwind[:, m:m + hp.in_seqlen, :, :],
             sshg[:, m:m + hp.in_seqlen, :, :], thflx[:, m:m + hp.in_seqlen, :, :]], (1, 2, 3, 4, 0))
        test_samples.append(data)
        pred_start = m + hp.in_seqlen - 1 + hp.lead_time
        sst_true.append(sst_origin[pred_start:pred_start + hp.out_seqlen, :, :])
        ssta_true.append(sst_origin[pred_start:pred_start + hp.out_seqlen, :, :] - np.expand_dims(cli_mean, axis=0))

    model = UConvlstm(hp)
    model.load_weights(f'{hp.delivery_model_dir}/{hp.delivery_model_file}')
    for x_in in test_samples:
        y_out = np.squeeze(model(x_in, training=False))  # (1, t, h, w, 1)-->(t, h, w)
        y_pred = np.reshape(sst_scaler.inverse_transform(np.reshape(y_out, (hp.out_seqlen, h * w))),
                            (hp.out_seqlen, h, w))
        sst_preds.append(y_pred)
        ssta_preds.append(y_pred - np.expand_dims(cli_mean, axis=0))

    return sst_true, sst_preds, ssta_true, ssta_preds


def array2seq(arr, lead):
    all_seq = []
    for n in range(0, len(arr), lead):
        all_seq.append(arr[n])
    # print(all_seq)
    merge_seq = np.concatenate(all_seq, axis=0)
    bias = len(arr) + hp.out_seqlen - len(merge_seq) - 1
    if bias != 0:
        all_seq.append(arr[len(arr) - 1][hp.out_seqlen - bias:])
        merge_seq = np.concatenate(all_seq, axis=0)
    print("index seq:", merge_seq.shape)
    return merge_seq


def seq_three_mon_avg(seq_index):
    r = seq_index.rolling(window=3, center=False, min_periods=1)
    results = r.mean()
    return results


def metric_nino():
    sst_true, sst_preds, ssta_true, ssta_preds = predict()
    nino_true = nino_seq(ssta_true)
    nino_preds = nino_seq(ssta_preds)
    rmse = get_rmse(nino_true, nino_preds)
    skill, cor = pcc(nino_true, nino_preds)
    return rmse, skill, cor


def nino_index_curve():
    sst_true, sst_preds, ssta_true, ssta_preds = predict()
    nino_true_seq = seq_three_mon_avg(array2seq(nino_seq(ssta_true), lead=hp.out_seqlen))
    nino_preds_seq = seq_three_mon_avg(array2seq(nino_seq(ssta_preds), lead=hp.out_seqlen))
    return nino_true_seq, nino_preds_seq


def predict_one_sample():
    h = hp.height
    w = hp.width
    in_scope = range(0, hp.in_seqlen)
    out_scope = range(hp.in_seqlen, hp.in_seqlen+hp.out_seqlen)

    sst, uwind, vwind, sshg, thflx, sst_origin = preprocess_helper(h, w, one_sample=True)
    x_in = np.transpose([sst[:, in_scope, :, :], uwind[:, in_scope, :, :], vwind[:, in_scope, :, :],
                         sshg[:, in_scope, :, :], thflx[:, in_scope, :, :]], (1, 2, 3, 4, 0))
    y_true = sst_origin[out_scope, :, :]
    cli_mean = np.mean(sst_origin, axis=0)
    ssta_true = y_true-np.expand_dims(cli_mean, axis=0)

    sst_scaler = MinMaxScaler()
    sst_scaler.fit_transform(np.reshape(sst_origin, (-1, h * w)))

    model = UConvlstm(hp)
    model.load_weights(f'{hp.delivery_model_dir}/{hp.delivery_model_file}')
    y_out = np.squeeze(model(x_in, training=False))  # (1, t, h, w, 1)-->(t, h, w)
    sst_pred = np.reshape(sst_scaler.inverse_transform(np.reshape(y_out, (hp.out_seqlen, h * w))),
                        (hp.out_seqlen, h, w))
    ssta_pred = sst_pred - np.expand_dims(cli_mean, axis=0)

    return ssta_true, ssta_pred


if __name__ == "__main__":
    ssta_true, ssta_pred = predict_one_sample()
    plot_helper(ssta_pred)
