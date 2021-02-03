# -*- coding: utf-8 -*-
# /usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer.

Building blocks for Transformer
'''

import numpy as np
import tensorflow as tf
import math


def ln(inputs, epsilon=1e-8, scope="ln"):
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def conv_attention_layer(inputs, k):
    outputs = tf.layers.dense(tf.layers.dense(inputs=inputs, units=k, activation='tanh'), units=1)
    outputs = tf.nn.softmax(outputs, -2)
    return outputs


def weighted_sum_block(info, alpha, time_len):
    info = tf.multiply(alpha, info)
    info = tf.keras.layers.Lambda(lambda x: tf.split(x, num_or_size_splits=time_len, axis=-2))(info)
    outputs = tf.keras.layers.add(info)
    return outputs


def mask(inputs, type=None):
    """Masks paddings on keys or queries to inputs
    inputs: 3d tensor. (h*N, T_q, T_k)
    key_masks: 3d tensor. (N, 1, T_k)
    type: string. "key" | "future"

    e.g.,
    >> inputs = tf.zeros([2, 2, 3], dtype=tf.float32)
    >> key_masks = tf.constant([[0., 0., 1.],
                                [0., 1., 1.]])
    >> mask(inputs, key_masks=key_masks, type="key")
    array([[[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],

       [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]],

       [[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],

       [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]]], dtype=float32)
    """
    padding_num = -2 ** 32 + 1
    # if type in ("k", "key", "keys"):
    #     key_masks = tf.to_float(key_masks)
    #     key_masks = tf.tile(key_masks, [tf.shape(inputs)[0] // tf.shape(key_masks)[0], 1])  # (h*N, seqlen)
    #     key_masks = tf.expand_dims(key_masks, 1)  # (h*N, 1, seqlen)
    #     outputs = inputs + key_masks * padding_num
    # elif type in ("q", "query", "queries"):
    #     # Generate masks
    #     masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
    #     masks = tf.expand_dims(masks, -1)  # (N, T_q, 1)
    #     masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])  # (N, T_q, T_k)
    #
    #     # Apply masks to inputs
    #     outputs = inputs*masks
    if type in ("f", "future", "right"):
        diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
        future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)

        paddings = tf.ones_like(future_masks) * padding_num
        outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
    else:
        print("Check if you entered type correctly!")

    return outputs


def cross_attention(Q, K, V, att_unit, causality=False, scope="cross_attention"):
    '''mask is applied before the softmax layer, no dropout is applied, '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # batch_size*num_heads
        segs = tf.shape(V)[0]
        time, measure = Q[0].get_shape().as_list()[1], Q[1].get_shape().as_list()[1]
        (value_units, Tunits, Munits) = att_unit

        (Q_T, Q_M) = Q
        (K_T, K_M) = K

        # Check the dimension consistency of the combined matrix
        assert Q_T.get_shape().as_list()[1:] == K_T.get_shape().as_list()[1:]
        assert Q_M.get_shape().as_list()[1:] == K_M.get_shape().as_list()[1:]
        assert Q_T.get_shape().as_list()[0] == Q_M.get_shape().as_list()[0]
        assert K_T.get_shape().as_list()[0] == K_M.get_shape().as_list()[0]

        # Build the Attention Map and scale
        AM_Time = tf.matmul(Q_T, tf.transpose(K_T, [0, 2, 1])) / tf.sqrt(tf.cast(Tunits, tf.float32))  # (N*h, T, T)
        AM_Measure = tf.matmul(Q_M, tf.transpose(K_M, [0, 2, 1])) / tf.sqrt(tf.cast(Tunits, tf.float32))  # (N*h, M, M)

        # causality or future blinding masking for decoder
        if causality:
            AM_Time = mask(AM_Time, type="future")

        AM_Time = tf.nn.softmax(AM_Time)
        AM_Measure = tf.nn.softmax(AM_Measure)

        shape_time = [segs, time, measure, value_units]
        shape_measure = [segs, measure, time, value_units]

        # decomposed manner in CDSA
        Out_Time = tf.reshape(tf.matmul(AM_Time, tf.reshape(V, [segs, time, measure*value_units])), shape_time)
        Out_Time = tf.transpose(Out_Time, perm=[0, 2, 1, 3])
        Out_Time_M = tf.reshape(tf.matmul(AM_Measure, tf.reshape(Out_Time, [segs, measure, time*value_units])), shape_measure)
    return tf.transpose(Out_Time_M, perm=[0, 2, 1, 3])


def multihead_attention(queries, keys, values, att_unit, value_attr,
                        num_heads=8,
                        dropout_rate=0,
                        training=True,
                        causality=False,
                        scope="multihead_attention"):
    '''Applies multihead attention. See 3.2.2
    queries: A 3d tensor with shape of [N, T_q, d_model].
    keys: A 3d tensor with shape of [N, T_k, d_model].
    values: A 3d tensor with shape of [N, T_k, d_model].
    att_unit:  the hyperparameter for the dimention of Q/K and V
    value_attr: kernel size and stride for conv_3d
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    num_heads: An int. Number of heads.
    dropout_rate: A floating point number.
    training: Boolean. Controller of mechanism for dropout.
    causality: Boolean. If true, units that reference the future are masked.
    scope: Optional scope for `variable_scope`.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    # assume input: [batch_size, time, measurement, feature]
    # d_model = queries.get_shape().as_list()[-1]
    (value_units, Tunits, Munits) = att_unit
    V_filters, (V_kernel, V_stride) = value_units * num_heads, value_attr
    batch = tf.shape(queries)[0]
    time, measure, feature = queries.get_shape().as_list()[1:]

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Linear projections
        Q_time = tf.layers.dense(tf.reshape(queries, [batch, time, measure*feature]), num_heads * Tunits, use_bias=True,
                                 name='Q_Time')
        K_time = tf.layers.dense(tf.reshape(keys, [batch, time, measure*feature]), num_heads * Tunits, use_bias=True, name='K_Time')
        Q_m = tf.layers.dense(tf.reshape(tf.transpose(queries, [0, 2, 1, 3]), [batch, measure, time*feature]),
                              num_heads * Munits, use_bias=True, name='Q_M')
        K_m = tf.layers.dense(tf.reshape(tf.transpose(keys, [0, 2, 1, 3]), [batch, measure, time*feature]), num_heads * Munits,
                              use_bias=True, name='K_M')
        # same shape with inputs [batch_size, time, measurement, feature]
        V = tf.layers.conv2d(inputs=values, filters=V_filters, kernel_size=V_kernel, strides=V_stride, padding='same',
                             data_format="channels_last", name='V')

        # Split and concat
        # Split the matrix to multiple heads and then concatenate to build a larger batch size:
        Qhb_time = tf.concat(tf.split(Q_time, num_heads, axis=2), axis=0)  # (h*N, T, Tunits)
        Khb_time = tf.concat(tf.split(K_time, num_heads, axis=2), axis=0)  # (h*N, T, Tunits)
        Qhb_m = tf.concat(tf.split(Q_m, num_heads, axis=2), axis=0)
        Khb_m = tf.concat(tf.split(K_m, num_heads, axis=2), axis=0)
        Q_headbatch = (Qhb_time, Qhb_m)
        K_headbatch = (Khb_time, Khb_m)

        # [batch_size*num_heads, time, measurement, value_units]
        V_headbatch = tf.concat(tf.split(V, num_heads, axis=3), axis=0)

        # Attention
        outputs = cross_attention(Q_headbatch, K_headbatch, V_headbatch, att_unit, causality)

        # Merge the multi-head back to the original shape
        # [batch_size, time, measurement, filters]
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=3)
        # outputs = tf.layers.dense(outputs, 1, name='multihead_fuse')
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = ln(outputs)

    return outputs


def ff(inputs, num_units, scope="positionwise_feedforward"):
    '''position-wise feed forward net. See 3.3
    
    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Inner layer
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

        # Outer layer
        outputs = tf.layers.dense(outputs, num_units[1])

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = ln(outputs)

    return outputs


def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
    inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
    epsilon: Smoothing rate.
    
    For example,
    
    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1], 
       [0, 1, 0],
       [1, 0, 0]],

      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)
       
    outputs = label_smoothing(inputs)
    
    with tf.Session() as sess:
        print(sess.run([outputs]))
    
    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],

       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]   
    ```    
    '''
    V = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / V)


def positional_encoding(inputs,
                        maxlen,
                        masking=True,
                        scope="positional_encoding"):
    '''Sinusoidal Positional_Encoding. See 3.5
    inputs: 3d tensor. (N, T, E)
    maxlen: scalar. Must be >= T
    masking: Boolean. If True, padding positions are set to zeros.
    scope: Optional scope for `variable_scope`.

    returns
    3d tensor that has the same shape as inputs.
    '''

    E = inputs.get_shape().as_list()[-1]  # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # (N, T)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i - i % 2) / E) for i in range(E)]
            for pos in range(maxlen)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, E)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)


def auxiliary_encode(inputs):
    # inputs: [batch_size, time, loc, measurement, 1]
    # outputs: [batch_size, time, loc, measurement, 1]
    # inputs: [b, t, m, f]
    # outputs: [b, t, m, f]
    N = tf.shape(inputs)[0]
    T, M, F = inputs.get_shape().as_list()[1:4]
    with tf.variable_scope('auxiliary_encode'):
        denom = tf.constant(1000.0)
        phase = tf.linspace(0.0, T - 1.0, T) * tf.constant(math.pi / 180.0) / denom
        sin = tf.expand_dims(tf.expand_dims(tf.sin(phase), 0), -1)
        time_encoding = tf.tile(tf.expand_dims(sin, -1), [N, 1, M, F])
        return time_encoding


def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)
