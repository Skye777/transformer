"""
@author: Skye Cui
@file: model.py
@time: 2021/2/23 15:55
@description: 
"""
import tensorflow as tf
from component.layers import Encoder, Decoder, ConvLstmBlock, EnConvlstm, DeConvlstm


class UTransformer(tf.keras.Model):
    def __init__(self, hp):
        super(UTransformer, self).__init__()
        self.hp = hp
        self.encoder = Encoder(num_layers=hp.num_blocks,
                               num_predictor=hp.num_predictor,
                               att_unit=(hp.vunits, hp.MTunits, hp.Tunits, hp.Munits),
                               value_attr=(hp.V_kernel, hp.V_stride),
                               in_seqlen=hp.in_seqlen,
                               num_heads=hp.num_heads,
                               model_structure=hp.model_structure,
                               d_ff=hp.d_ff,
                               d_model=hp.d_model,
                               drop_rate=hp.dropout_rate)

        self.decoder = Decoder(num_layers=hp.num_blocks,
                               num_predictor=hp.num_predictor,
                               att_unit=(hp.vunits, hp.MTunits, hp.Tunits, hp.Munits),
                               value_attr=(hp.V_kernel, hp.V_stride),
                               out_seqlen=hp.out_seqlen,
                               num_heads=hp.num_heads,
                               model_structure=hp.model_structure,
                               d_ff=hp.d_ff,
                               d_model=hp.d_model,
                               drop_rate=hp.dropout_rate)

    def call(self, inputs, training=None, mask=None):
        # print("inputs:", inp)
        x, ys = inputs
        enc_output, skip_layers = self.encoder(x, training)

        if training:
            tar, dec_inp = ys
            dec_output = self.decoder([dec_inp, enc_output, skip_layers, self.hp.out_seqlen], training=True)

        else:
            decoder_inp_start = tf.expand_dims(x[:, -1, :, :, :], 1)
            decoder_inputs = decoder_inp_start

            for i in range(self.hp.out_seqlen):
                dec_out = self.decoder([decoder_inputs, enc_output, skip_layers, i+1], training=False)
                decoder_inputs = tf.concat((decoder_inp_start, dec_out), 1)

            dec_output = decoder_inputs[:, 1:, :, :, :]

        return dec_output


class StackConvlstm(tf.keras.Model):
    def __init__(self, hp):
        super(StackConvlstm, self).__init__()
        self.hp = hp
        self.stackConvlstm = ConvLstmBlock(hp)

    def call(self, inputs, training=None, mask=None):
        outputs = self.stackConvlstm(inputs, training)

        return outputs


class UConvlstm(tf.keras.Model):
    def __init__(self, hp):
        super(UConvlstm, self).__init__()
        self.encoder = EnConvlstm(seq_len=hp.in_seqlen)
        self.decoder = DeConvlstm(strategy=hp.strategy, out_seqlen=hp.out_seqlen)

    def call(self, inputs, training=None, mask=None):
        hidden_states, skip_layers = self.encoder(inputs, training)
        outputs = self.decoder([hidden_states, skip_layers], training)
        return outputs
