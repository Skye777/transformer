# -*- coding: utf-8 -*-
# /usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer

Note.
if safe, entities on the source side have the prefix 1, and the target side 2, for convenience.
For example, fpath1, fpath2 means source file path and target file path, respectively.
'''
import tensorflow as tf
from utils import calc_num_batches
import numpy as np


def load_vocab(vocab_fpath):
    '''Loads vocabulary file and returns idx<->token maps
    vocab_fpath: string. vocabulary file path.
    Note that these are reserved
    0: <pad>, 1: <unk>, 2: <s>, 3: </s>

    Returns
    two dictionaries.
    '''
    vocab = [line.split()[0] for line in open(vocab_fpath, 'r', encoding='utf-8').read().splitlines()]
    token2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2token = {idx: token for idx, token in enumerate(vocab)}
    return token2idx, idx2token


def load_data(fpath1, fpath2, maxlen1, maxlen2):
    '''Loads source and target data and filters out too lengthy samples.
    fpath1: source file path. string.
    fpath2: target file path. string.
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.

    Returns
    sents1: list of source sents
    sents2: list of target sents
    '''
    sents1, sents2 = [], []
    with open(fpath1, 'r', encoding='utf-8') as f1, open(fpath2, 'r', encoding='utf-8') as f2:
        for sent1, sent2 in zip(f1, f2):
            if len(sent1.split()) + 1 > maxlen1: continue  # 1: </s>
            if len(sent2.split()) + 1 > maxlen2: continue  # 1: </s>
            sents1.append(sent1.strip())
            sents2.append(sent2.strip())
    return sents1, sents2


def encode(inp, type, dict):
    '''Converts string to number. Used for `generator_fn`.
    inp: 1d byte array.
    type: "x" (source side) or "y" (target side)
    dict: token2idx dictionary

    Returns
    list of numbers
    '''
    inp_str = inp.decode("utf-8")
    if type == "x":
        tokens = inp_str.split() + ["</s>"]
    else:
        tokens = ["<s>"] + inp_str.split() + ["</s>"]

    x = [dict.get(t, dict["<unk>"]) for t in tokens]
    return x


def generator_fn():
    '''Generates training / evaluation data
    sents1: list of source sents
    sents2: list of target sents
    vocab_fpath: string. vocabulary file path.

    yields
    xs: tuple of
        x: list of source token ids in a sent
        x_seqlen: int. sequence length of x
        sent1: str. raw source (=input) sentence
    labels: tuple of
        decoder_input: decoder_input: list of encoded decoder inputs
        y: list of target token ids in a sent
        y_seqlen: int. sequence length of y
        sent2: str. target sentence
    '''
    train_x, train_y = np.random.rand(10, 3, 320, 640, 3), np.random.rand(10, 3, 320, 640, 3)
    for x, y in zip(train_x, train_y):
        yield x, (y, y)


def input_fn(batch_size, shuffle=False):
    '''Batchify data
    sents1: list of source sents
    sents2: list of target sents
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean

    Returns
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    '''
    types = (tf.float32,
             (tf.float32, tf.float32))

    dataset = tf.data.Dataset.from_generator(generator_fn,
                                             output_types=types,
                                             args=())  # <- arguments for generator_fn. converted to np string arrays

    if shuffle:  # for training
        dataset = dataset.shuffle(128 * batch_size)

    dataset = dataset.repeat()  # iterate forever
    dataset = dataset.batch(batch_size).prefetch(1)

    return dataset


def get_batch(batch_size, shuffle=False):
    '''Gets training / evaluation mini-batches
    fpath1: source file path. string.
    fpath2: target file path. string.
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean

    Returns
    batches
    num_batches: number of mini-batches
    num_samples
    '''
    # sents1, sents2 = load_data(fpath1, fpath2, maxlen1, maxlen2)
    batches = input_fn(batch_size, shuffle=shuffle)
    num_batches = calc_num_batches(10, batch_size)
    return batches, num_batches, 10


if __name__ == '__main__':
    generator_fn()
