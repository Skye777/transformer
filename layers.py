"""
@author: Skye Cui
@file: layers.py
@time: 2021/2/22 13:43
@description: 
"""
import tensorflow as tf
from modules import cross_attention, ff, auxiliary_encode


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_predictor, att_unit, value_attr, in_seqlen, num_heads, model_structure, d_ff, d_model, drop_rate):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.in_seqlen = in_seqlen

        self.embedding = Embedding(num_predictor, seq_len=in_seqlen, encode_phase=True)
        self.enc_layers = [
            EncoderLayer(att_unit, value_attr, num_heads, model_structure, d_ff, d_model, drop_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training):
        x, skip_layers = self.embedding(x)
        x += auxiliary_encode(x, T=self.in_seqlen)
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training)

        return x, skip_layers


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_predictor, att_unit, value_attr, out_seqlen, num_heads, model_structure, d_ff, d_model, drop_rate):
        super(Decoder, self).__init__()
        self.num_layers = num_layers

        self.embedding = Embedding(num_predictor, seq_len=out_seqlen, encode_phase=False)
        self.dec_layers = [
            DecoderLayer(att_unit, value_attr, num_heads, model_structure, d_ff, d_model, drop_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)
        self.restore = Restore(num_predictor)

    def call(self, x, enc_output, skip_layers, seq_len, training):
        x, _ = self.embedding(x)
        if training is False:
            print(x.shape)

        x += auxiliary_encode(x, T=seq_len)
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, training)
        x = self.restore(x, skip_layers)

        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, att_unit, value_attr, num_heads, model_structure, d_ff, d_model, drop_rate):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(att_unit, value_attr, num_heads, model_structure, causality=False)
        self.ffn = ff(num_units=[d_ff, d_model])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training):
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, att_unit, value_attr, num_heads, model_structure, d_ff, d_model, drop_rate):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(att_unit, value_attr, num_heads, model_structure, causality=True)
        self.mha2 = MultiHeadAttention(att_unit, value_attr, num_heads, model_structure, causality=False)

        self.ffn = ff(num_units=[d_ff, d_model])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, enc_output, training):
        attn1 = self.mha1(x, x, x)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2 = self.mha2(out1, enc_output, enc_output)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, att_unit,
                 value_attr,
                 num_heads,
                 model_structure,
                 causality):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.att_unit = att_unit
        (self.value_units, self.MT_units, self.Tunits, self.Munits) = att_unit
        self.V_filters, (self.V_kernel, self.V_stride) = self.value_units * num_heads, value_attr
        self.model_structure = model_structure
        self.causality = causality

        self.qt = tf.keras.layers.Dense(self.num_heads * self.Tunits)
        self.kt = tf.keras.layers.Dense(self.num_heads * self.Tunits)
        self.qm = tf.keras.layers.Dense(self.num_heads * self.Munits)
        self.km = tf.keras.layers.Dense(self.num_heads * self.Munits)
        self.v = tf.keras.layers.Conv2D(filters=self.V_filters, kernel_size=self.V_kernel, strides=self.V_stride,
                                        padding='same', data_format="channels_last")
        self.qmt = tf.keras.layers.Dense(self.num_heads * self.MT_units)
        self.kmt = tf.keras.layers.Dense(self.num_heads * self.MT_units)
        self.vmt = tf.keras.layers.Conv2D(filters=self.V_filters, kernel_size=self.V_kernel, strides=self.V_stride,
                                          padding='same', data_format="channels_last")
        self.dense = tf.keras.layers.Dense(self.V_filters)

    def call(self, queries, keys, values):
        batch = tf.shape(queries)[0]
        q_t, k_t = queries.get_shape().as_list()[1], keys.get_shape().as_list()[1]
        measure, feature = queries.get_shape().as_list()[2:]
        if self.model_structure == 'Decomposed':
            # Linear projections
            Q_time = self.qt(tf.reshape(queries, [batch, q_t, measure * feature]))
            K_time = self.kt(tf.reshape(keys, [batch, k_t, measure * feature]))
            Q_m = self.qm(tf.reshape(tf.transpose(queries, [0, 2, 1, 3]), [batch, measure, q_t * feature]))
            K_m = self.km(tf.reshape(tf.transpose(keys, [0, 2, 1, 3]), [batch, measure, k_t * feature]))
            # same shape with inputs [batch_size, time, measurement, feature]
            V = self.v(values)

            # Split and concat
            # Split the matrix to multiple heads and then concatenate to build a larger batch size:
            Qhb_time = tf.concat(tf.split(Q_time, self.num_heads, axis=2), axis=0)  # (h*N, T, Tunits)
            Khb_time = tf.concat(tf.split(K_time, self.num_heads, axis=2), axis=0)  # (h*N, T, Tunits)
            Qhb_m = tf.concat(tf.split(Q_m, self.num_heads, axis=2), axis=0)
            Khb_m = tf.concat(tf.split(K_m, self.num_heads, axis=2), axis=0)
            Q_headbatch = (Qhb_time, Qhb_m)
            K_headbatch = (Khb_time, Khb_m)

            # [batch_size*num_heads, time, measurement, value_units]
            V_headbatch = tf.concat(tf.split(V, self.num_heads, axis=3), axis=0)

        else:
            Q = self.qmt(tf.reshape(queries, [batch, q_t * measure, feature]))
            K = self.kmt(tf.reshape(keys, [batch, k_t * measure, feature]))
            # same shape with inputs [batch_size, time, measurement, feature]
            V = self.vmt(values)

            Q_headbatch = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0)
            K_headbatch = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0)
            V_headbatch = tf.concat(tf.split(V, self.num_heads, axis=3), axis=0)
        # Attention
        outputs = cross_attention(Q_headbatch, K_headbatch, V_headbatch, self.att_unit, q_t, k_t, self.model_structure,
                                  self.causality)

        # Merge the multi-head back to the original shape
        # [batch_size, time, measurement, filters]
        outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=3)
        outputs = self.dense(outputs)

        return outputs


class Embedding(tf.keras.layers.Layer):
    def __init__(self, num_predictor, seq_len, encode_phase):
        super(Embedding, self).__init__()
        self.num_predictor = num_predictor
        self.encode_phase = encode_phase

        self.convblock1 = ConvMaxPoolBlock(filters=4, kernel_size=3, pool_size=2, strides=2, t=seq_len, h=80, w=160, c=4) # here t used only in encoder
        self.convblock2 = ConvMaxPoolBlock(filters=8, kernel_size=3, pool_size=2, strides=2, t=seq_len, h=40, w=80, c=8)
        self.convblock3 = ConvMaxPoolBlock(filters=16, kernel_size=3, pool_size=2, strides=2, t=seq_len, h=20, w=40, c=16)

        self.conv2d = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=5, activation=tf.keras.layers.LeakyReLU()))

    def call(self, inputs):
        # inputs: [batch_size, time, h, w, predictor]
        T = tf.shape(inputs)[1]
        inputs = tf.expand_dims(tf.transpose(inputs, [4, 0, 1, 2, 3]), -1)  # (predictor, batch, time, w, h, 1)
        embeddings = []
        skip_layers = {}

        for i in range(self.num_predictor):
            if self.encode_phase:
                out, map1 = self.convblock1(inputs[i], True)
                out, map2 = self.convblock2(out, True)
                out, map3 = self.convblock3(out, True)
                skip_layers['predictor{}_map1'.format(i+1)] = map1
                skip_layers['predictor{}_map2'.format(i+1)] = map2
                skip_layers['predictor{}_map3'.format(i+1)] = map3
            else:
                out = self.convblock1(inputs[i], False)
                out = self.convblock2(out, False)
                out = self.convblock3(out, False)
            out = tf.keras.layers.Reshape((T, 8 * 4 * 32))(self.conv2d(out))
            embeddings.append(out)
        embeddings = tf.transpose(embeddings, [1, 2, 0, 3])  # (b, t, m, f)

        return embeddings, skip_layers


class Restore(tf.keras.layers.Layer):
    def __init__(self, num_predictor):
        super(Restore, self).__init__()
        self.num_predictor = num_predictor

        self.deconv = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=5, strides=5, activation=tf.keras.layers.LeakyReLU()))
        self.bn = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())

        self.deconvblock1 = ConvTransBlock(filters=8, kernel_size=3, up_size=2)
        self.deconvblock2 = ConvTransBlock(filters=4, kernel_size=3, up_size=2)
        self.deconvblock3 = ConvTransBlock(filters=1, kernel_size=3, up_size=2)

    def call(self, inputs, skip_layers):
        # assume inputs: (b, t, m, d_model)
        T = tf.shape(inputs)[1]
        inputs = tf.keras.layers.Reshape((T, 4, 8, 32, self.num_predictor))(inputs)
        inputs = tf.transpose(inputs, [5, 0, 1, 2, 3, 4])
        outputs = []

        for i in range(self.num_predictor):
            deconv_out = self.bn(self.deconv(inputs[i]))
            out = self.deconvblock1(deconv_out, skip_layers['predictor{}_map3'.format(i+1)])
            out = self.deconvblock2(out, skip_layers['predictor{}_map2'.format(i+1)])
            out = self.deconvblock3(out, skip_layers['predictor{}_map1'.format(i+1)])
            outputs.append(out)
        outputs = tf.keras.layers.Concatenate()(outputs)
        return outputs


class ConvTransBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, up_size):
        super(ConvTransBlock, self).__init__()
        self.deconv = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, padding='same',
                                            activation=tf.keras.layers.LeakyReLU()))
        self.up_sampling = tf.keras.layers.TimeDistributed(tf.keras.layers.UpSampling2D(size=up_size))
        self.bn = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())

    def call(self, inputs, map):
        T = tf.shape(inputs)[1]

        skip_layer = tf.tile(tf.expand_dims(map, 1), [1, T, 1, 1, 1])
        inputs = tf.keras.layers.Concatenate()([skip_layer, inputs])

        deconv_out = self.deconv(inputs)
        up_sampling_out = self.up_sampling(deconv_out)
        bn_out = self.bn(up_sampling_out)
        return bn_out


class ConvMaxPoolBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, pool_size, strides, t, h, w, c):
        super(ConvMaxPoolBlock, self).__init__()
        self.conv = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                                   activation=tf.keras.layers.LeakyReLU()))
        self.max_pool = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=strides))
        self.bn = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())
        self.alpha = ConvAttention(t, h, w, c, k=16)
        self.get_feature_maps = WeightedSumBlock(t, h, w, c)

    def call(self, inputs, encode_phase):
        conv_out = self.conv(inputs)
        pool_out = self.max_pool(conv_out)
        bn_out = self.bn(pool_out)
        if encode_phase:
            alpha = self.alpha(bn_out)
            skip_layer_feature_map = self.get_feature_maps(bn_out, alpha)
            return bn_out, skip_layer_feature_map
        else:
            return bn_out


class ConvAttention(tf.keras.layers.Layer):
    def __init__(self, l, h, w, c, k):
        super(ConvAttention, self).__init__()
        self.reshape = tf.keras.layers.Reshape((l, w*h*c))
        self.layer1 = tf.keras.layers.Dense(units=k, activation='tanh')
        self.layer2 = tf.keras.layers.Dense(units=1)

    def call(self, inputs):
        outputs = self.layer2(self.layer1(self.reshape(inputs)))
        outputs = tf.nn.softmax(outputs, axis=-2)
        return outputs


class WeightedSumBlock(tf.keras.layers.Layer):
    def __init__(self, l, h, w, c):
        super(WeightedSumBlock, self).__init__()
        self.l = l
        self.add = tf.keras.layers.Add()
        self.reshape1 = tf.keras.layers.Reshape((l, w*h*c))
        self.reshape2 = tf.keras.layers.Reshape((h, w, c))

    def call(self, inputs, alpha):
        inputs = self.reshape1(inputs)
        info = tf.multiply(alpha, inputs)
        info = tf.keras.layers.Lambda(lambda x: tf.split(x, num_or_size_splits=self.l, axis=-2))(info)
        outputs = tf.keras.layers.add(info)
        outputs = self.reshape2(outputs)
        return outputs
