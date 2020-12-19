import numpy as np
import os
import csv
import random
import tensorflow as tf


def load_seismic_sample(_DataPath):
    obj_dir = os.listdir(_DataPath)
    #_file_num = len(obj_dir)
    i = 0
    _sample_num = 0
    for file in obj_dir:
        if file.endswith(".csv"):
            with open(_DataPath+file, 'r', newline='') as h:
                rd = csv.reader(h, dialect='excel', quoting=csv.QUOTE_NONNUMERIC)
                result_read = np.array(list(rd))
                [_time_len, _sensor_num] = result_read.shape
                seismic_array = result_read.reshape((1, _time_len, _sensor_num))
                if i == 0:
                    ret = seismic_array
                else:
                    ret = np.concatenate((ret, seismic_array), axis = 0)
                _sample_num = _sample_num + 1
        else:
            pass
        i = i + 1
    return ret

def tf_log_10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype = numerator.dtype))
  return numerator / denominator

def coefficient_correlation(_series_1, _series_2):
    shape_1 = _series_1.get_shape().as_list()
    shape_2 = _series_2.get_shape().as_list()
    Covariance = tf.reduce_mean(tf.multiply(_series_1 - tf.reduce_mean(_series_1)*tf.ones(shape_1,dtype=tf.float32),
                                            _series_2 - tf.reduce_mean(_series_2)*tf.ones(shape_2,dtype=tf.float32)))
    variance_1 = tf.reduce_mean(tf.square(_series_1 - tf.reduce_mean(_series_1)*tf.ones(shape_1,dtype=tf.float32)))
    variance_2 = tf.reduce_mean(tf.square(_series_2 - tf.reduce_mean(_series_2)*tf.ones(shape_2,dtype=tf.float32)))
    ret_coef = Covariance/tf.sqrt(variance_1*variance_2)
    return ret_coef

def structure_similar(_image_1, _image_2):
    shape = _image_1.get_shape().as_list()
    Covariance = tf.reduce_mean(tf.multiply(_image_1 - tf.reduce_mean(_image_1)*tf.ones(shape,dtype=tf.float32),
                                            _image_2 - tf.reduce_mean(_image_2)*tf.ones(shape,dtype=tf.float32)))
    variance_1 = tf.reduce_mean(tf.square(_image_1 - tf.reduce_mean(_image_1)*tf.ones(shape,dtype=tf.float32)))
    variance_2 = tf.reduce_mean(tf.square(_image_2 - tf.reduce_mean(_image_2)*tf.ones(shape,dtype=tf.float32)))
    ret_ssv = (Covariance/tf.sqrt(variance_1*variance_2))+(2*tf.sqrt(variance_1*variance_2)/(variance_1+variance_2))
    return ret_ssv/2

def weighted_SNR_func(image_ori, image_rec):
    shape = image_ori.get_shape().as_list()
    signal = tf.reduce_mean(tf.reduce_mean(tf.square(image_ori),axis=3),axis=0)
    recovery = tf.reduce_mean(tf.reduce_mean(tf.square(image_rec),axis=3),axis=0)
    signal_sum = tf.reduce_sum(tf.reduce_mean(tf.reduce_mean(tf.square(image_ori),axis=3),axis=0),axis=1)
    signal_loss = tf.reduce_sum(tf.reduce_mean(tf.reduce_mean(tf.square(image_rec - image_ori),axis=3),axis=0),axis=1)
    signal_snr_pertime = tf.divide(signal_loss, signal_sum)
    weights = tf.zeros([shape[1]],dtype=tf.float32)
    for i in range(shape[1]):
        corvariance = tf.reduce_mean(tf.multiply(signal[i,:] - tf.reduce_mean(signal[i,:])*tf.ones([1, shape[2]],dtype=tf.float32),
                                                recovery[i,:] - tf.reduce_mean(recovery[i,:])*tf.ones([1, shape[2]],dtype=tf.float32)))
        varicance_ori = tf.reduce_mean(tf.square(signal[i,:] - tf.reduce_mean(signal[i,:])*tf.ones([1, shape[2]],dtype=tf.float32)))
        varicance_rec = tf.reduce_mean(tf.square(recovery[i,:] - tf.reduce_mean(recovery[i,:])*tf.ones([1, shape[2]],dtype=tf.float32)))
        one_hot = tf.one_hot(i,shape[1],dtype=tf.float32)
        weights_value = 1 - corvariance/tf.sqrt(varicance_ori*varicance_rec)
        weights = weights + weights_value*one_hot
    weights = weights/tf.reduce_sum(weights)
    weighted_SNR_val = tf.reduce_sum(tf.multiply(weights, signal_snr_pertime))
    return weighted_SNR_val

def transform_measuring_mtx(_A_val, _in_height):
    _in_wideth, _cs_num = _A_val.shape
    _A_temp = np.zeros((_in_wideth*_in_height,_cs_num*_in_height), dtype=float)
    for i in range(_in_height):
        for j in range(_cs_num):
            _A_temp[i:i+_in_wideth,i*j] = _A_val[:,j]
    _A = tf.constant(_A_temp, dtype=tf.float32)
    return _A

def transform_input(_y_batch, _batch_num, _cs_num):
    _input_temp = tf.reshape(_y_batch, (_batch_num, -1, _cs_num))
    _shape = _input_temp.get_shape().as_list()
    _input = tf.reshape(_input_temp, (_shape[0],_shape[1],_shape[2],1))
    return _input

class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                        decay=self.momentum,
                        updates_collections=None,
                        epsilon=self.epsilon,
                        scale=True,
                        is_training=train,
                        scope=self.name)

def linear(input_, output_size_, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size_], tf.float32,
                        tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size_],
                        initializer=tf.constant_initializer(bias_start))
    if with_w:
        return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
        return tf.matmul(input_, matrix) + bias

def trans_fc2d(input_, output_size_, stddev=0.02, name="fc2d", with_w=False):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(name):
        w_1 = tf.get_variable('w_1', [1, 1, shape[3], 2*output_size_],
                    initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv_1 = tf.nn.conv2d(input_, w_1, strides=[1, 1, 1, 1], padding='SAME')
        biases_1 = tf.get_variable('biases_1', [2*output_size_], initializer=tf.constant_initializer(0.0))
        fc_layer_1 = tf.nn.bias_add(conv_1, biases_1)

        w_2 = tf.get_variable('w_2', [1, 1, 2*output_size_, output_size_],
                    initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv_2 = tf.nn.conv2d(fc_layer_1, w_2, strides=[1, 1, 1, 1], padding='SAME')
        biases_2 = tf.get_variable('biases_2', [output_size_], initializer=tf.constant_initializer(0.0))
        fc_layer_2 = tf.nn.bias_add(conv_2, biases_2)

        conv = tf.reshape(fc_layer_2, fc_layer_2.get_shape())
        re_conv = tf.transpose(conv, perm=[0, 1, 3, 2])
    if with_w:
        return re_conv, w, biases
    else:
        return re_conv

def conv2d(input_, output_dim, k_h, k_w, d_h=1, d_w=1, stddev=0.02, name="conv2d", with_w=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                    initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        if with_w:
            return conv, w, biases
        else:
            return conv

def deconv2d(input_, output_shape, k_h, k_w, d_h=1, d_w=1, stddev=0.02, name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                      initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                      strides=[1, d_h, d_w, 1], padding="SAME")
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, w, biases
        else:
            return deconv

def get_learning_rate(global_step, learning_rate , _Decay_lr_Iter):
    if _Decay_lr_Iter > 0:           # If it is positive.
        return tf.train.exponential_decay(learning_rate, global_step, _Decay_lr_Iter, 0.9, staircase=True)
    else:
        return tf.constant(learning_rate)
