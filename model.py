# -*- coding: utf-8 -*-
# /usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer

Transformer network
'''
import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed, Reshape, LeakyReLU, Concatenate, UpSampling2D
import numpy as np

from data_load import load_vocab
from modules import ff, conv_attention_layer, auxiliary_encode, multihead_attention, label_smoothing, noam_scheme, \
    weighted_sum_block
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)


class Transformer:
    '''
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    training: boolean.
    '''

    def __init__(self, hp):
        self.hp = hp
        self.token2idx, self.idx2token = load_vocab(hp.vocab)
        self.skip_layer_feature_maps_1 = []
        self.skip_layer_feature_maps_2 = []
        self.skip_layer_feature_maps_3 = []
        self.skip_layer_feature_maps_4 = []

    def embedding_module(self, inputs, encode_phase):
        # inputs: [batch_size, time, w, h, predictor]
        T = tf.shape(inputs)[1]
        predictors = self.hp.num_predictor
        inputs = tf.expand_dims(tf.transpose(inputs, [4, 0, 1, 2, 3]), -1)   # (predictor, batch, time, w, h, 1)
        embeddings = []

        for i in range(predictors):
            # inputs = Input(shape=(time, 320, 640, 1))
            conv_out = TimeDistributed(tf.layers.Conv2D(filters=2, kernel_size=3, padding='same', activation=LeakyReLU()),
                                       input_shape=(T, 320, 640, 1))(inputs[i])  # (b, t, 320, 640, f1)
            pool_out = TimeDistributed(tf.layers.MaxPooling2D(pool_size=2, strides=2))(conv_out)  # (b, t, 160, 320, f1)
            bn_out = TimeDistributed(tf.layers.BatchNormalization())(pool_out)
            if encode_phase:
                alpha = conv_attention_layer(Reshape((T, 160*320*2))(bn_out), k=16)
                self.skip_layer_feature_maps_1.append(Reshape((160, 320, 2))(
                    weighted_sum_block(info=Reshape((T, 160 * 320 * 2))(bn_out), alpha=alpha, time_len=self.hp.in_seqlen))) # (b, 160, 320, f1)

            conv_out = TimeDistributed(tf.layers.Conv2D(filters=4, kernel_size=3, padding='same', activation=LeakyReLU()))(bn_out)
            pool_out = TimeDistributed(tf.layers.MaxPooling2D(pool_size=2, strides=2))(conv_out)   # (b, t, 80, 160, f2)
            bn_out = TimeDistributed(tf.layers.BatchNormalization())(pool_out)
            if encode_phase:
                alpha = conv_attention_layer(Reshape((T, 80*160*4))(bn_out), k=16)
                self.skip_layer_feature_maps_2.append(Reshape((80, 160, 4))(
                        weighted_sum_block(info=Reshape((T, 80 * 160 * 4))(bn_out), alpha=alpha, time_len=self.hp.in_seqlen))) # (b, 80, 160, f2)

            conv_out = TimeDistributed(tf.layers.Conv2D(filters=8, kernel_size=3, padding='same', activation=LeakyReLU()))(bn_out)
            pool_out = TimeDistributed(tf.layers.MaxPooling2D(pool_size=2, strides=2))(conv_out)  # (b, t, 40, 80, f3)
            bn_out = TimeDistributed(tf.layers.BatchNormalization())(pool_out)
            if encode_phase:
                alpha = conv_attention_layer(Reshape((T, 40*80*8))(bn_out), k=16)
                self.skip_layer_feature_maps_3.append(Reshape((40, 80, 8))(
                        weighted_sum_block(info=Reshape((T, 40 * 80 * 8))(bn_out), alpha=alpha, time_len=self.hp.in_seqlen)))  # (b, 40, 80, f3)

            conv_out = TimeDistributed(tf.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation=LeakyReLU()))(bn_out)
            pool_out = TimeDistributed(tf.layers.MaxPooling2D(pool_size=2, strides=2))(conv_out)   # (b, t, 20, 40, f4)
            bn_out = TimeDistributed(tf.layers.BatchNormalization())(pool_out)
            if encode_phase:
                alpha = conv_attention_layer(Reshape((T, 20*40*16))(bn_out), k=16)
                self.skip_layer_feature_maps_4.append(Reshape((20, 40, 16))(
                        weighted_sum_block(info=Reshape((T, 20 * 40 * 16))(bn_out), alpha=alpha, time_len=self.hp.in_seqlen)))  # (b, 20, 40, f4)

            out = TimeDistributed(tf.layers.Conv2D(filters=32, kernel_size=5, strides=5, activation=LeakyReLU()))(bn_out)  # (b, t, 4, 8, f5)
            out_feature = Reshape((T, 4*8*32))(out)  # (b, t, 4*8*f3)
            embeddings.append(out_feature)
        embeddings = tf.transpose(embeddings, [1, 2, 0, 3])   # (b, t, m, f)
        return embeddings

    def restore_module(self, inputs):
        # assume inputs: (b, t, m, d_model)
        T = tf.shape(inputs)[1]
        predictors = self.hp.num_predictor
        inputs = Reshape((T, 4, 8, 32, predictors))(inputs)
        inputs = tf.transpose(inputs, [5, 0, 1, 2, 3, 4])
        outputs = []

        for i in range(predictors):
            # inputs = Input(shape=(time, 4, 8, 32))
            deconv_out = TimeDistributed(tf.layers.Conv2DTranspose(filters=16, kernel_size=5, strides=5, activation=LeakyReLU()))(inputs=inputs[i])
            bn_out = TimeDistributed(tf.layers.BatchNormalization())(deconv_out)
            skip_layer_4 = tf.tile(tf.expand_dims(self.skip_layer_feature_maps_4[i], 1), [1, T, 1, 1, 1])
            deconv_out = Concatenate()([skip_layer_4, bn_out])   # (b, t, 20, 40, f4)

            deconv_out = TimeDistributed(tf.layers.Conv2DTranspose(filters=8, kernel_size=3, padding='same', activation=LeakyReLU()))(deconv_out)
            up_sampling_out = TimeDistributed(UpSampling2D(size=2))(deconv_out)
            bn_out = TimeDistributed(tf.layers.BatchNormalization())(up_sampling_out)
            skip_layer_3 = tf.tile(tf.expand_dims(self.skip_layer_feature_maps_3[i], 1), [1, T, 1, 1, 1])
            deconv_out = Concatenate()([skip_layer_3, bn_out])   # (b, t, 40, 80, f3)

            deconv_out = TimeDistributed(tf.layers.Conv2DTranspose(filters=4, kernel_size=3, padding='same', activation=LeakyReLU()))(deconv_out)
            up_sampling_out = TimeDistributed(UpSampling2D(size=2))(deconv_out)
            bn_out = TimeDistributed(tf.layers.BatchNormalization())(up_sampling_out)
            skip_layer_2 = tf.tile(tf.expand_dims(self.skip_layer_feature_maps_2[i], 1), [1, T, 1, 1, 1])
            deconv_out = Concatenate()([skip_layer_2, bn_out])   # (b, t, 80, 160, f2)

            deconv_out = TimeDistributed(tf.layers.Conv2DTranspose(filters=2, kernel_size=3, padding='same', activation=LeakyReLU()))(deconv_out)
            up_sampling_out = TimeDistributed(UpSampling2D(size=2))(deconv_out)
            bn_out = TimeDistributed(tf.layers.BatchNormalization())(up_sampling_out)
            skip_layer_1 = tf.tile(tf.expand_dims(self.skip_layer_feature_maps_1[i], 1), [1, T, 1, 1, 1])
            deconv_out = Concatenate()([skip_layer_1, bn_out])  # (b, t, 160, 320, f1)

            deconv_out = TimeDistributed(tf.layers.Conv2DTranspose(filters=1, kernel_size=3, padding='same', activation=LeakyReLU()))(deconv_out)
            up_sampling_out = TimeDistributed(UpSampling2D(size=2))(deconv_out)
            bn_out = TimeDistributed(tf.layers.BatchNormalization())(up_sampling_out)  # (b, t, 320, 640, 1)
            outputs.append(bn_out)

        outputs = Concatenate()(outputs)
        return outputs

    def encode(self, x, training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            # assume x: [batch_size, time, width, height, measurement]
            # embedding
            enc = self.embedding_module(x, encode_phase=True)   # [b, t, m, f]
            # position encoding
            enc += auxiliary_encode(enc, T=self.hp.in_seqlen)
            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)

            # Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              att_unit=(self.hp.vunits, self.hp.MTunits, self.hp.Tunits, self.hp.Munits),
                                              value_attr=(self.hp.V_kernel, self.hp.V_stride),
                                              time=self.hp.in_seqlen,
                                              num_heads=self.hp.num_heads,
                                              model_structure=self.hp.model_structure,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model])
        memory = enc
        return memory

    def decode(self, ys, memory, T, training=True):
        '''
        memory: encoder outputs. (N, T1, d_model)
        src_masks: (N, T1)

        Returns
        logits: (N, T2, V). float32.
        y_hat: (N, T2). int32
        y: (N, T2). int32
        sents2: (N,). string.
        '''
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            # assume decoder_inputs: [batch_size, time, width, height, measurement]
            # assume y(label): [batch_size, time, width, height, measurement] all predictors
            decoder_inputs, y = ys

            # embedding
            dec = self.embedding_module(decoder_inputs, encode_phase=False)  # [b, t, m, f]

            # position encoding
            dec += auxiliary_encode(dec, T)
            dec = tf.layers.dropout(dec, self.hp.dropout_rate, training=training)

            # Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # Masked self-attention (Note that causality is True at this time)
                    dec = multihead_attention(queries=dec,
                                              keys=dec,
                                              values=dec,
                                              att_unit=(self.hp.vunits, self.hp.MTunits, self.hp.Tunits, self.hp.Munits),
                                              value_attr=(self.hp.V_kernel, self.hp.V_stride),
                                              time=T,
                                              num_heads=self.hp.num_heads,
                                              model_structure=self.hp.model_structure,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=True,
                                              scope="self_attention")

                    # Vanilla attention
                    dec = multihead_attention(queries=dec,
                                              keys=memory,
                                              values=memory,
                                              att_unit=(self.hp.vunits, self.hp.MTunits, self.hp.Tunits, self.hp.Munits),
                                              value_attr=(self.hp.V_kernel, self.hp.V_stride),
                                              time=T,
                                              num_heads=self.hp.num_heads,
                                              model_structure=self.hp.model_structure,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False,
                                              scope="vanilla_attention")
                    # Feed Forward
                    dec = ff(dec, num_units=[self.hp.d_ff, self.hp.d_model])  # (b, t, m, d_model)
            y_hat = self.restore_module(dec)   # (b, t, w, h, f)

        return y_hat, y

    def train(self, x, ys):
        '''
        Returns
        loss: scalar.
        train_op: training operation
        global_step: scalar.
        summaries: training summary node
        '''
        # forward
        # inputs = Input(shape=(12, 320, 640, 1))
        memory = self.encode(x)
        preds, y = self.decode(ys, memory, T=self.hp.out_seqlen)   # (b, t, w, h, m)

        # train scheme
        loss = tf.reduce_mean(tf.square(y - preds))

        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, summaries

    def eval(self, xs, ys):
        '''Predicts autoregressively
        At inference, input ys is ignored.
        Returns
        y_hat: (N, T2)
        '''
        decoder_inputs, y = ys

        decoder_inputs = tf.ones((tf.shape(xs[0])[0], 1, self.hp.height, self.hp.width, self.hp.num_predictor), tf.float32)
        ys = (decoder_inputs, y)

        memory = self.encode(xs, training=False)

        logging.info("Inference graph is being built. Please be patient.")
        for i in tqdm(range(self.hp.out_seqlen)):
            y_hat, y = self.decode(ys, memory, T=i+1, training=False)
            _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
            ys = (_decoder_inputs, y)

        # # monitor a random sample
        # n = tf.random_uniform((), 0, tf.shape(y_hat)[0] - 1, tf.int32)
        # sent1 = sents1[n]
        # pred = convert_idx_to_token_tensor(y_hat[n], self.idx2token)
        # sent2 = sents2[n]

        # tf.summary.text("sent1", sent1)
        # tf.summary.text("pred", pred)
        # tf.summary.text("sent2", sent2)
        summaries = tf.summary.merge_all()

        return y_hat, summaries


if __name__ == '__main__':
    from hparams import Hparams

    x, y = tf.convert_to_tensor(np.random.rand(4, 12, 320, 640, 3)), tf.convert_to_tensor(np.random.rand(4, 12, 320, 640, 3))
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    ys = (y, y)
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()
    m = Transformer(hp)
    m.train(x, ys)
