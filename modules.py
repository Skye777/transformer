"""
@author: Skye Cui
@file: modules.py
@time: 2021/2/24 15:04
@description: 
"""
import tensorflow as tf
import math


def mask(inputs):
    padding_num = -2 ** 32 + 1

    diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
    tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
    future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)

    paddings = tf.ones_like(future_masks) * padding_num
    outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)

    return outputs


def cross_attention(Q, K, V, att_unit, q_t, k_t, model_structure, causality=False):
    '''mask is applied before the softmax layer, no dropout is applied, '''
    # batch_size*num_heads
    segs = tf.shape(V)[0]
    (value_units, MT_units, Tunits, Munits) = att_unit
    measure = V.get_shape().as_list()[2]

    shape_time = [segs, q_t, measure, value_units]
    shape_measure = [segs, measure, q_t, value_units]

    if model_structure == 'Decomposed':
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
            AM_Time = mask(AM_Time)

        AM_Time = tf.nn.softmax(AM_Time)
        AM_Measure = tf.nn.softmax(AM_Measure)

        # decomposed manner in CDSA
        Out_Time = tf.reshape(tf.matmul(AM_Time, tf.reshape(V, [segs, time, measure*value_units])), shape_time)
        Out_Time = tf.transpose(Out_Time, perm=[0, 2, 1, 3])
        Out_Time_M = tf.reshape(tf.matmul(AM_Measure, tf.reshape(Out_Time, [segs, measure, time*value_units])), shape_measure)
        Outputs = tf.transpose(Out_Time_M, perm=[0, 2, 1, 3])

    else:
        AM = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / tf.sqrt(tf.cast(MT_units, tf.float32))  # (N*h, T*M, T*M)

        # causality or future blinding masking for decoder
        if causality:
            AM = mask(AM)

        AM = tf.nn.softmax(AM)
        Outputs = tf.reshape(tf.matmul(AM, tf.reshape(V, [segs, k_t*measure, value_units])), shape_time)

    return Outputs


def auxiliary_encode(inputs, T):
    # inputs: [batch_size, time, loc, measurement, 1]
    # outputs: [batch_size, time, loc, measurement, 1]
    # inputs: [b, t, m, f]
    # outputs: [b, t, m, f]
    N = tf.shape(inputs)[0]
    M, F = inputs.get_shape().as_list()[2:4]
    denom = tf.constant(1000.0)
    phase = tf.linspace(0.0, T - 1.0, T) * tf.constant(math.pi / 180.0) / denom
    sin = tf.expand_dims(tf.expand_dims(tf.sin(phase), 0), -1)
    time_encoding = tf.tile(tf.expand_dims(sin, -1), [N, 1, M, F])
    return time_encoding


def ff(num_units):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(num_units[0], activation='relu'),  # (batch_size, seq_len, .., dff)
        tf.keras.layers.Dense(num_units[1])  # (batch_size, seq_len, .., d_model)
    ])
