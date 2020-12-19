import numpy as np
import os
import csv
import time
import tensorflow as tf
import math
import psutil
import random

from csgan_utils import *

Save_Times = 1
CS_Number_Ratio = 16
SNR_Cpiont = 50

Recov_Channel = 8         # recovery's convolutional channel dimension
DisMeas_Channel = 8       # Measurement Discriminator's convolutional channel dimension
DisReco_Channel = 8       # Recovery Discriminator's convolutional channel dimension

Epoch_num = 5000
enhance_value = 0.99
cur_epoch = 0
LearningRate = 0.01        # 0.01~1.0
MinConInd = 0.01
global_step = 0

Discriminator_Class = 3    # the number of classfications to be learned by discriminator
Weights_factor = 0.3
Weights_vector = -1 - np.array(range(Discriminator_Class-1), dtype=np.float32)
Class_Weights_pre = (Weights_vector/sum(abs(Weights_vector)))
for i in range(Discriminator_Class-1):
    if Class_Weights_pre[i] > 0:
        Class_Weights_pre[i] = Class_Weights_pre[i]*Weights_factor
Class_Weights = Class_Weights_pre.reshape((Discriminator_Class-1,1))
print(Class_Weights)

class CSGAN(object):
    def __init__(self, input_height, input_width, batch_size, cs_num,
                 Re_channel, DM_channel, DR_channel):
        """The constructor of CSGAN class. An object containing a computation
         graph corresponding to a loaded config is created."""
        self.input_height = input_height
        self.input_width = input_width
        self.batch_size = batch_size
        self.cs_num = cs_num
        self.Re_channel = Re_channel  # Recovery's convolutional channel dimension.
        self.DM_channel = DM_channel  # Measurment Discriminator's convolutional channel dimension.
        self.DR_channel = DR_channel  # Recovery Discriminator's convolutional channel dimension.

        self.True_Label = tf.one_hot(tf.zeros([batch_size],dtype=tf.int32), Discriminator_Class)

        self.build_model()

    def build_model(self):
        """Creates the appropriate TensorFlow computation graph."""
        self.image_dims = [self.input_height, self.input_width, 1]
        # Placeholder for compressed measurements.
        self.xs_target = tf.placeholder(tf.float32, [self.batch_size] + self.image_dims, name='target_xs')
        # Placeholder for generator inputs
        # Batch normalization (deals with poor initialization helps gradient flow).
        # conv+BN+relu
        self.Re_BN0_3_2 = batch_norm(name='re_bn0_3_2')
        self.Re_BN0_4_2 = batch_norm(name='re_bn0_4_2')
        self.Re_BN0_4_3 = batch_norm(name='re_bn0_4_3')
        self.Re_BN0_5_2 = batch_norm(name='re_bn0_5_2')
        self.Re_BN0_5_3 = batch_norm(name='re_bn0_5_3')
        self.Re_BN0 = batch_norm(name='re_bn0')
        self.Re_BN1_1 = batch_norm(name='re_bn1_1')
        self.Re_BN1_2 = batch_norm(name='re_bn1_2')
        self.Re_BN1_3_1 = batch_norm(name='re_bn1_3_1')
        self.Re_BN1_3_2 = batch_norm(name='re_bn1_3_2')
        self.Re_BN1_4_1 = batch_norm(name='re_bn1_4_1')
        self.Re_BN1_4_2 = batch_norm(name='re_bn1_4_2')
        self.Re_BN1_4_3 = batch_norm(name='re_bn1_4_3')
        self.Re_BN1_5_1 = batch_norm(name='re_bn1_5_1')
        self.Re_BN1_5_2 = batch_norm(name='re_bn1_5_2')
        self.Re_BN1_5_3 = batch_norm(name='re_bn1_5_3')
        self.Re_BN1 = batch_norm(name='re_bn1')
        self.Re_BN2_1 = batch_norm(name='re_bn2_1')
        self.Re_BN2_2 = batch_norm(name='re_bn2_2')
        self.Re_BN2_3_1 = batch_norm(name='re_bn2_3_1')
        self.Re_BN2_3_2 = batch_norm(name='re_bn2_3_2')
        self.Re_BN2_4_1 = batch_norm(name='re_bn2_4_1')
        self.Re_BN2_4_2 = batch_norm(name='re_bn2_4_2')
        self.Re_BN2_4_3 = batch_norm(name='re_bn2_4_3')
        self.Re_BN2 = batch_norm(name='re_bn2')
        self.Re_BN3_1 = batch_norm(name='re_bn3_1')
        self.Re_BN3_2 = batch_norm(name='re_bn3_2')
        self.Re_BN3_3_1 = batch_norm(name='re_bn3_3_1')
        self.Re_BN3_3_2 = batch_norm(name='re_bn3_3_2')
        self.Re_BN3 = batch_norm(name='re_bn3')
        self.Re_BN4_1 = batch_norm(name='re_bn4_1')
        self.Re_BN4_2 = batch_norm(name='re_bn4_2')
        self.Re_BN4 = batch_norm(name='re_bn4')
        self.Re_BN5_1 = batch_norm(name='re_bn5_1')
        self.Re_BN5_2 = batch_norm(name='re_bn5_2')
        self.Re_BN5 = batch_norm(name='re_bn5')

        self.d_SR_bn1 = batch_norm(name='d_sr_bn1')
        self.d_SR_bn2 = batch_norm(name='d_sr_bn2')
        self.d_SR_bn3 = batch_norm(name='d_sr_bn3')
        self.d_SR_bn4 = batch_norm(name='d_sr_bn4')
        self.d_SR_bn5 = batch_norm(name='d_sr_bn5')
        self.d_SR_bn6 = batch_norm(name='d_sr_bn6')
        self.d_SR_bn7 = batch_norm(name='d_sr_bn7')

        self.d_Re_bn0 = batch_norm(name='d_Re_bn0')
        self.d_Re_bn1_1 = batch_norm(name='d_Re_bn1_1')
        self.d_Re_bn1_2 = batch_norm(name='d_Re_bn1_2')
        self.d_Re_bn1 = batch_norm(name='d_Re_bn1')
        self.d_Re_bn2_1 = batch_norm(name='d_Re_bn2_1')
        self.d_Re_bn2_2 = batch_norm(name='d_Re_bn2_2')
        self.d_Re_bn2 = batch_norm(name='d_Re_bn2')
        self.d_Re_bn3_1 = batch_norm(name='d_Re_bn3_1')
        self.d_Re_bn3_2 = batch_norm(name='d_Re_bn3_2')
        self.d_Re_bn3 = batch_norm(name='d_Re_bn3')
        self.d_Re_bn4_1 = batch_norm(name='d_Re_bn4_1')
        self.d_Re_bn4_2 = batch_norm(name='d_Re_bn4_2')
        self.d_Re_bn4 = batch_norm(name='d_Re_bn4')
        self.d_Re_bn5 = batch_norm(name='d_Re_bn5')
        self.d_Re_bn6 = batch_norm(name='d_Re_bn6')
        self.d_Re_bn7 = batch_norm(name='d_Re_bn7')
        self.d_Re_bn8 = batch_norm(name='d_Re_bn8')
        self.d_Re_bn9 = batch_norm(name='d_Re_bn9')
        # Loss for optimizers.
        self.M_opt_Loss, self.M_opt_obver_SNR, self.M_opt_recov_SNR, self.M_adver_loss, self.R_adver_loss = self.measurement_opt()
        self.discr_opt_Loss, self.discr_real_loss, self.discr_fake_loss = self.discr_measure_opt()
        self.IR_opt_Loss, self.Int_Recov_SNR, self.IR_adver_Loss, self.IR_percept_Loss = self.integ_recovery_opt()
        self.disc_re_opt_Loss, self.disc_re_real_loss, self.disc_re_fake_loss, self.discr_Loss_sum, self.disc_recov_loss = self.discr_recovery_opt()
        self.saver = tf.train.Saver()

    def measurement_process(self, input, with_w = False, train = True):
        input_ = tf.transpose(input, perm=[0, 1, 3, 2])
        with tf.variable_scope("Measurement_FC", reuse=tf.AUTO_REUSE):
            w = tf.get_variable('w', [1, 1, self.input_width, self.cs_num],
                        initializer=tf.truncated_normal_initializer(stddev = 1.0))
            conv = tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', [self.cs_num], initializer=tf.constant_initializer(0.0))
            fc_layer = tf.nn.bias_add(conv, biases)
            ret_mat = tf.transpose(fc_layer, perm=[0, 1, 3, 2])
        if with_w:
            return ret_mat, w
        else:
            return ret_mat

    def postprocess(self, input, batch_size=None, train=True):
        input_ = tf.reshape(input,[self.batch_size, self.input_height, 1, -1])
        In_shape = input_.get_shape().as_list()
        with tf.variable_scope("inte_recov_FC", reuse=tf.AUTO_REUSE):
            w_1 = tf.get_variable('w_1', [1, 1, In_shape[-1], self.input_width],
                        initializer=tf.truncated_normal_initializer(stddev = 0.02))
            conv_1 = tf.nn.conv2d(input_, w_1, strides=[1, 1, 1, 1], padding='SAME')
            biases_1 = tf.get_variable('biases_1', [self.input_width], initializer=tf.constant_initializer(0.0))
            fc_layer_1 = tf.nn.bias_add(conv_1, biases_1)
            ret_fc = tf.transpose(fc_layer_1, perm=[0, 1, 3, 2])
        return ret_fc

    def integ_recovery(self, input, batch_size=None, train=True):
        with tf.variable_scope("inte_recov_CNN", reuse=tf.AUTO_REUSE):
            #inception 1
            #h0_1 = tf.nn.leaky_relu(conv2d(input, self.Re_channel*1, k_h=1, k_w=1, name='g_sr_h0_1_conv'))
            h0_2 = tf.nn.leaky_relu(conv2d(input, self.Re_channel*1, k_h=3, k_w=3, name='g_sr_h0_2_conv'))
            h0_3_1 = tf.nn.leaky_relu(conv2d(input, self.Re_channel*1, k_h=3, k_w=3, name='g_sr_h0_3_1_conv'))
            h0_3_2 = tf.nn.leaky_relu(self.Re_BN0_3_2(conv2d(h0_3_1, self.Re_channel*1, k_h=3, k_w=3, name='g_sr_h0_3_2_conv'), train=train))
            h0_4_1 = tf.nn.leaky_relu(conv2d(input, self.Re_channel*1, k_h=5, k_w=5, name='g_sr_h0_4_1_conv'))
            h0_4_2 = tf.nn.leaky_relu(self.Re_BN0_4_2(conv2d(h0_4_1, self.Re_channel*1, k_h=3, k_w=3, name='g_sr_h0_4_2_conv'), train=train))
            h0_4_3 = tf.nn.leaky_relu(self.Re_BN0_4_3(conv2d(h0_4_2, self.Re_channel*1, k_h=3, k_w=3, name='g_sr_h0_4_3_conv'), train=train))
            h0_5_1 = tf.nn.leaky_relu(conv2d(input, self.Re_channel*1, k_h=5, k_w=5, name='g_sr_h0_5_1_conv'))
            h0_5_2 = tf.nn.leaky_relu(self.Re_BN0_5_2(conv2d(h0_5_1, self.Re_channel*1, k_h=5, k_w=5, name='g_sr_h0_5_2_conv'), train=train))
            h0_5_3 = tf.nn.leaky_relu(self.Re_BN0_5_3(conv2d(h0_5_2, self.Re_channel*1, k_h=3, k_w=3, name='g_sr_h0_5_3_conv'), train=train))
            h0_concat = tf.concat([h0_2, h0_3_2, h0_4_3, h0_5_3], 3)
            h0 = tf.nn.leaky_relu(self.Re_BN0(conv2d(h0_concat, self.Re_channel*4, k_h=1, k_w=1, name='g_sr_h0_conv'), train=train)) + h0_concat
            #inception 2
            #h1_1 = tf.nn.leaky_relu(self.Re_BN1_1(conv2d(h0, self.Re_channel*2, k_h=1, k_w=1, name='g_sr_h1_1_conv'), train=train))
            h1_2 = tf.nn.leaky_relu(self.Re_BN1_2(conv2d(h0, self.Re_channel*2, k_h=3, k_w=3, name='g_sr_h1_2_conv'), train=train))
            h1_3_1 = tf.nn.leaky_relu(self.Re_BN1_3_1(conv2d(h0, self.Re_channel*2, k_h=3, k_w=3, name='g_sr_h1_3_1_conv'), train=train))
            h1_3_2 = tf.nn.leaky_relu(self.Re_BN1_3_2(conv2d(h1_3_1, self.Re_channel*2, k_h=3, k_w=3, name='g_sr_h1_3_2_conv'), train=train))
            h1_4_1 = tf.nn.leaky_relu(self.Re_BN1_4_1(conv2d(h0, self.Re_channel*2, k_h=3, k_w=3, name='g_sr_h1_4_1_conv'), train=train))
            h1_4_2 = tf.nn.leaky_relu(self.Re_BN1_4_2(conv2d(h1_4_1, self.Re_channel*2, k_h=3, k_w=3, name='g_sr_h1_4_2_conv'), train=train))
            h1_4_3 = tf.nn.leaky_relu(self.Re_BN1_4_3(conv2d(h1_4_2, self.Re_channel*2, k_h=3, k_w=3, name='g_sr_h1_4_3_conv'), train=train))
            h1_5_1 = tf.nn.leaky_relu(self.Re_BN1_5_1(conv2d(h0, self.Re_channel*2, k_h=5, k_w=5, name='g_sr_h1_5_1_conv'), train=train))
            h1_5_2 = tf.nn.leaky_relu(self.Re_BN1_5_2(conv2d(h1_5_1, self.Re_channel*2, k_h=3, k_w=3, name='g_sr_h1_5_2_conv'), train=train))
            h1_5_3 = tf.nn.leaky_relu(self.Re_BN1_5_3(conv2d(h1_5_2, self.Re_channel*2, k_h=3, k_w=3, name='g_sr_h1_5_3_conv'), train=train))
            h1_concat = tf.concat([h1_2, h1_3_2, h1_4_3, h1_5_3], 3)
            h1 = tf.nn.leaky_relu(self.Re_BN1(conv2d(h1_concat, self.Re_channel*8, k_h=1, k_w=1, name='g_sr_h1_conv'), train=train))+h1_concat
            #inception 3
            #h2_1 = tf.nn.leaky_relu(self.Re_BN2_1(conv2d(h1, self.Re_channel*4, k_h=1, k_w=1, name='g_sr_h2_1_conv'), train=train))
            h2_2 = tf.nn.leaky_relu(self.Re_BN2_2(conv2d(h1, self.Re_channel*4, k_h=3, k_w=3, name='g_sr_h2_2_conv'), train=train))
            h2_3_1 = tf.nn.leaky_relu(self.Re_BN2_3_1(conv2d(h1, self.Re_channel*4, k_h=3, k_w=3, name='g_sr_h2_3_1_conv'), train=train))
            h2_3_2 = tf.nn.leaky_relu(self.Re_BN2_3_2(conv2d(h2_3_1, self.Re_channel*4, k_h=3, k_w=3, name='g_sr_h2_3_2_conv'), train=train))
            h2_4_1 = tf.nn.leaky_relu(self.Re_BN2_4_1(conv2d(h1, self.Re_channel*4, k_h=3, k_w=3, name='g_sr_h2_4_1_conv'), train=train))
            h2_4_2 = tf.nn.leaky_relu(self.Re_BN2_4_2(conv2d(h2_4_1, self.Re_channel*4, k_h=3, k_w=3, name='g_sr_h2_4_2_conv'), train=train))
            h2_4_3 = tf.nn.leaky_relu(self.Re_BN2_4_3(conv2d(h2_4_2, self.Re_channel*4, k_h=3, k_w=3, name='g_sr_h2_4_3_conv'), train=train))
            h2_concat = tf.concat([h2_2, h2_3_2, h2_4_3], 3)
            h2 = tf.nn.leaky_relu(self.Re_BN2(conv2d(h2_concat, self.Re_channel*12, k_h=1, k_w=1, name='g_sr_h2_conv'), train=train))+h2_concat
            #inception 4
            #h3_1 = tf.nn.leaky_relu(self.Re_BN3_1(conv2d(h2, self.Re_channel*8, k_h=1, k_w=1, name='g_sr_h3_1_conv'), train=train))
            h3_2 = tf.nn.leaky_relu(self.Re_BN3_2(conv2d(h2, self.Re_channel*8, k_h=3, k_w=3, name='g_sr_h3_2_conv'), train=train))
            h3_3_1 = tf.nn.leaky_relu(self.Re_BN3_3_1(conv2d(h2, self.Re_channel*8, k_h=3, k_w=3, name='g_sr_h3_3_1_conv'), train=train))
            h3_3_2 = tf.nn.leaky_relu(self.Re_BN3_3_2(conv2d(h3_3_1, self.Re_channel*8, k_h=3, k_w=3, name='g_sr_h3_3_2_conv'), train=train))
            h3_concat = tf.concat([h3_2, h3_3_2], 3)
            h3 = tf.nn.leaky_relu(self.Re_BN3(conv2d(h3_concat, self.Re_channel*16, k_h=1, k_w=1, name='g_sr_h3_conv'), train=train))+h3_concat
            #inception 5
            h4_1 = tf.nn.leaky_relu(self.Re_BN4_1(conv2d(h3, self.Re_channel*16, k_h=1, k_w=1, name='g_sr_h4_1_conv'), train=train))
            h4_2 = tf.nn.leaky_relu(self.Re_BN4_2(conv2d(h3, self.Re_channel*16, k_h=3, k_w=3, name='g_sr_h4_2_conv'), train=train))
            h4_concat = tf.concat([h4_1, h4_2], 3)
            h4 = tf.nn.leaky_relu(self.Re_BN4(conv2d(h4_concat, self.Re_channel*32, k_h=1, k_w=1, name='g_sr_h4_conv'), train=train))+h4_concat
            #inception 6
            h5_1 = tf.nn.leaky_relu(self.Re_BN5_1(conv2d(h4, self.Re_channel*32, k_h=1, k_w=1, name='g_sr_h5_1_conv'), train=train))+h4
            h5_2 = tf.nn.leaky_relu(self.Re_BN5_2(conv2d(h4, self.Re_channel*32, k_h=3, k_w=3, name='g_sr_h5_2_conv'), train=train))+h4
            h5_concat = tf.concat([h5_1, h5_2], 3)
            h5 = tf.nn.leaky_relu(self.Re_BN5(conv2d(h5_concat, self.Re_channel*64, k_h=1, k_w=1, name='g_sr_h5_conv'), train=train))+h5_concat
        return h5
#反向传播，获得测量矩阵的具体元素
    def discriminator_for_measure(self,representaion, batch_size=None, train=True):
        with tf.variable_scope("disc_mr_CNN", reuse=tf.AUTO_REUSE):
            h0 = tf.nn.relu(conv2d(representaion, self.DM_channel, k_h=7, k_w=7, d_h=2, d_w=2, name='d_sr_h0_conv'))
            h1 = tf.nn.relu(self.d_SR_bn1(conv2d(h0, self.DM_channel*2, k_h=5, k_w=5, d_h=2, d_w=2, name='d_sr_h1_conv'), train=train))
            h2 = tf.nn.relu(self.d_SR_bn2(conv2d(h1, self.DM_channel*4, k_h=5, k_w=5, d_h=1, d_w=1, name='d_sr_h2_conv'), train=train))
            h3 = tf.nn.relu(self.d_SR_bn3(conv2d(h2, self.DM_channel*8, k_h=3, k_w=3, d_h=2, d_w=2, name='d_sr_h3_conv'), train=train))
            h4 = tf.nn.relu(self.d_SR_bn4(conv2d(h3, self.DM_channel*8, k_h=3, k_w=3, d_h=1, d_w=1, name='d_sr_h4_conv'), train=train))
            h5 = tf.nn.relu(self.d_SR_bn5(conv2d(h4, self.DM_channel*16, k_h=3, k_w=3, d_h=1, d_w=1, name='d_sr_h5_conv'), train=train))
            flaten_data = tf.reshape(h5,[self.batch_size, -1])
            fc_1 = tf.nn.relu(self.d_SR_bn6(linear(flaten_data, self.DM_channel*64, scope="d_fc_1"), train=train))
            fc_2 = tf.nn.relu(self.d_SR_bn7(linear(fc_1, self.DM_channel*8, scope="d_fc_2"), train=train))
            classfication = tf.sigmoid(linear(fc_2, 1, scope="d_fc_3"))
        return classfication
#正向传播，由重构关系求得恢复的地震数据
    def discriminator_for_recovery(self,representaion, batch_size=None, train=True):
        with tf.variable_scope("disc_rec_CNN", reuse=tf.AUTO_REUSE):
            h0_1 = tf.nn.relu(conv2d(representaion, self.DR_channel*1, k_h=7, k_w=7, d_h=2, d_w=2, name='d_rec_h0_1_conv'))
            h0_2 = tf.nn.relu(conv2d(representaion, self.DR_channel*1, k_h=5, k_w=5, d_h=2, d_w=2, name='d_rec_h0_2_conv'))
            h0_s = tf.concat([h0_1, h0_2], 3)
            h0 = tf.nn.relu(self.d_Re_bn0(conv2d(h0_s, self.DR_channel*2, k_h=1, k_w=1, d_h=1, d_w=1, name='d_rec_h0_conv'), train=train))+h0_s

            h1_1 = tf.nn.relu(self.d_Re_bn1_1(conv2d(h0, self.DR_channel*2, k_h=5, k_w=5, d_h=2, d_w=2, name='d_rec_h1_1_conv'), train=train))
            h1_2 = tf.nn.relu(self.d_Re_bn1_2(conv2d(h0, self.DR_channel*2, k_h=3, k_w=3, d_h=2, d_w=2, name='d_rec_h1_2_conv'), train=train))
            h1_s = tf.concat([h1_1, h1_2], 3)
            h1 = tf.nn.relu(self.d_Re_bn1(conv2d(h1_s, self.DR_channel*4, k_h=1, k_w=1, d_h=1, d_w=1, name='d_rec_h1_conv'), train=train))+h1_s

            h2_1 = tf.nn.relu(self.d_Re_bn2_1(conv2d(h1, self.DR_channel*4, k_h=5, k_w=5, d_h=2, d_w=2, name='d_rec_h2_1_conv'), train=train))
            h2_2 = tf.nn.relu(self.d_Re_bn2_2(conv2d(h1, self.DR_channel*4, k_h=3, k_w=3, d_h=2, d_w=2, name='d_rec_h2_2_conv'), train=train))
            h2_s = tf.concat([h2_1, h2_2], 3)
            h2 = tf.nn.relu(self.d_Re_bn2(conv2d(h2_s, self.DR_channel*8, k_h=1, k_w=1, d_h=1, d_w=1, name='d_rec_h2_conv'), train=train))+h2_s

            h3_1 = tf.nn.relu(self.d_Re_bn3_1(conv2d(h2, self.DR_channel*8, k_h=5, k_w=5, d_h=2, d_w=2, name='d_rec_h3_1_conv'), train=train))
            h3_2 = tf.nn.relu(self.d_Re_bn3_2(conv2d(h2, self.DR_channel*8, k_h=3, k_w=3, d_h=2, d_w=2, name='d_rec_h3_2_conv'), train=train))
            h3_s = tf.concat([h3_1, h3_2], 3)
            h3 = tf.nn.relu(self.d_Re_bn3(conv2d(h3_s, self.DR_channel*16, k_h=1, k_w=1, d_h=1, d_w=1, name='d_rec_h3_conv'), train=train))+h3_s

            h4_1 = tf.nn.relu(self.d_Re_bn4_1(conv2d(h3, self.DR_channel*16, k_h=5, k_w=5, d_h=2, d_w=2, name='d_rec_h4_1_conv'), train=train))
            h4_2 = tf.nn.relu(self.d_Re_bn4_2(conv2d(h3, self.DR_channel*16, k_h=3, k_w=3, d_h=2, d_w=2, name='d_rec_h4_2_conv'), train=train))
            h4_s = tf.concat([h4_1, h4_2], 3)
            h4 = tf.nn.relu(self.d_Re_bn4(conv2d(h4_s, self.DR_channel*32, k_h=1, k_w=1, d_h=1, d_w=1, name='d_rec_h4_conv'), train=train))+h4_s

            h5 = tf.nn.relu(self.d_Re_bn5(conv2d(h4, self.DR_channel*64, k_h=3, k_w=3, d_h=1, d_w=1, name='d_rec_h5_conv'), train=train))
            percption_layer_0 = tf.reshape(h0,[self.batch_size, -1])
            percption_layer_1 = tf.reshape(h1,[self.batch_size, -1])
            percption_layer_2 = tf.reshape(h2,[self.batch_size, -1])
            percption_layer_3 = tf.reshape(h3,[self.batch_size, -1])
            percption_layer_4 = tf.reshape(h4,[self.batch_size, -1])
            percption_layer_5 = tf.reshape(h5,[self.batch_size, -1])
            discr_perceptual = tf.concat([percption_layer_0, percption_layer_2, percption_layer_4, percption_layer_5], 1)
            flaten_data = tf.concat([percption_layer_5], 1)
            fc_1 = tf.nn.relu(self.d_Re_bn8(linear(flaten_data, self.DR_channel*64, scope="d_fc_1"), train=train))
            fc_2 = tf.nn.relu(self.d_Re_bn9(linear(fc_1, self.DR_channel*8, scope="d_fc_2"), train=train))
            classfication = tf.nn.softmax(linear(fc_2, Discriminator_Class, scope="d_fc_3"))
            #classfication_pre = tf.sigmoid(linear(fc_2, Discriminator_Class, scope="d_fc_3"))
            #classfication = classfication_pre/tf.tile(tf.reshape(tf.reduce_sum(classfication_pre,1),[self.batch_size,1]),[1,Discriminator_Class])
        return classfication, discr_perceptual

    def measurement_opt(self):
        M_inputs = self.measurement_process(self.xs_target)
        M_recover_pre = self.integ_recovery(M_inputs, batch_size=self.batch_size)
        M_recover = self.postprocess(M_recover_pre, batch_size=self.batch_size)
        M_recover_SNR_n = tf.reduce_sum(tf.reduce_mean(tf.square(M_recover - self.xs_target),axis=0))/tf.reduce_sum(tf.reduce_mean(tf.square(self.xs_target),axis=0))
        M_recover_SNR = tf_log_10(M_recover_SNR_n)
        M_gen_inputs = self.measurement_process(M_recover)
        obser_SNR_n = tf.reduce_sum(tf.reduce_mean(tf.square(M_inputs - M_gen_inputs),axis=0))/tf.reduce_sum(tf.reduce_mean(tf.square(M_inputs),axis=0))
        obser_SNR = tf_log_10(obser_SNR_n)
        adver_loss = tf.reduce_mean(tf.square(self.discriminator_for_measure(M_gen_inputs)-tf.ones([self.batch_size,1],dtype=tf.float32)))
        discr_value, percept_value = self.discriminator_for_recovery(M_recover)
        adver_loss_Re_temp = tf.abs(discr_value - self.True_Label)
        adver_loss_Re = tf.reduce_mean(adver_loss_Re_temp[:,0])
        #weighted_SNR_Re = weighted_SNR_func(self.xs_target, M_recover)
        #weighted_SNR_Me = weighted_SNR_func(M_inputs, M_gen_inputs)
        M_opt_Loss = adver_loss + adver_loss_Re + obser_SNR_n + M_recover_SNR_n
        return M_opt_Loss, obser_SNR, M_recover_SNR, tf.sqrt(adver_loss), tf.sqrt(adver_loss_Re)

    def discr_measure_opt(self):
        SR_inputs = self.measurement_process(self.xs_target)
        SR_recover_pre = self.integ_recovery(SR_inputs, batch_size=self.batch_size)
        SR_discr_recover = self.postprocess(SR_recover_pre, batch_size=self.batch_size)
        discr_inputs = self.measurement_process(SR_discr_recover)
        real_loss = tf.square(self.discriminator_for_measure(SR_inputs)-tf.ones([self.batch_size,1],dtype=tf.float32))
        fake_loss = tf.square(self.discriminator_for_measure(discr_inputs))
        discr_SR_Loss = tf.reduce_mean(real_loss + fake_loss)
        return tf.sqrt(discr_SR_Loss/2), tf.sqrt(tf.reduce_mean(real_loss)), tf.sqrt(tf.reduce_mean(fake_loss))

    def integ_recovery_opt(self):
        Re_inputs = self.measurement_process(self.xs_target)
        Re_recover_pre = self.integ_recovery(Re_inputs, batch_size=self.batch_size)
        Re_gen_recover = self.postprocess(Re_recover_pre, batch_size=self.batch_size)
        Re_recovery_loss_seq = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.square(self.xs_target - Re_gen_recover),axis=3),axis=2),axis=1)
        Re_signal_seq = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.square(self.xs_target),axis=3),axis=2),axis=1)
        gen_recover_SNR_n = tf.reduce_mean(tf.divide(Re_recovery_loss_seq,Re_signal_seq))
        gen_recover_SNR = tf_log_10(gen_recover_SNR_n)
        Ratio_Linear = gen_recover_SNR_n/(10**(0-SNR_Cpiont/10)*tf.log(tf.constant(10,dtype=tf.float32))) - 1/tf.log(tf.constant(10,dtype=tf.float32)) - SNR_Cpiont/10
        def SNR_logri(): return gen_recover_SNR
        def SNR_linear(): return Ratio_Linear
        gen_recover_SNR_Cvlaue = tf.cond(tf.less(gen_recover_SNR, 0-SNR_Cpiont/10), SNR_linear, SNR_logri)
        discr_value_ori, percept_value_ori = self.discriminator_for_recovery(self.xs_target)
        discr_value, percept_value = self.discriminator_for_recovery(Re_gen_recover)
        adver_loss_Re_temp = tf.abs(discr_value - self.True_Label)
        adver_loss_Re = 1 - tf.reduce_mean(discr_value[:,0] + tf.matmul(discr_value[:,1:], Class_Weights))
        adver_loss_x = tf.reduce_mean(adver_loss_Re_temp[:,0])
        #adver_loss_Re = tf.reduce_mean(0.7*adver_loss_Re_temp[:,1]+1.3*adver_loss_Re_temp[:,2])
        percept_loss = tf.reduce_sum(tf.reduce_mean(tf.square(percept_value_ori - percept_value),axis=0))/tf.reduce_sum(tf.reduce_mean(tf.square(percept_value_ori),axis=0))
        #structure_sim = structure_similar(self.xs_target, Re_gen_recover)
        #weighted_SNR = weighted_SNR_func(self.xs_target, Re_gen_recover)
        diff_ssv = structure_similar(self.xs_target - Re_gen_recover, self.xs_target)
        #adver_loss_coeff = 1+tf.maximum(0.0,tf.reduce_mean(adver_loss_x)-0.5)*5
        def adver_loss_reduce(): return 1+tf.maximum(0.0,(tf.reduce_mean(adver_loss_x)-0.5)*tf.pow((1+tf.reduce_mean(adver_loss_Re)),2))
        def adver_loss_enlarge(): return tf.maximum(0.0,1-tf.maximum(0.0,0.5-tf.reduce_mean(adver_loss_x))*0)
        adver_loss_coeff = tf.cond(tf.less(tf.reduce_mean(adver_loss_x), 0.5), adver_loss_enlarge, adver_loss_reduce)
        IR_opt_Loss = gen_recover_SNR_Cvlaue + diff_ssv + adver_loss_coeff*adver_loss_Re + percept_loss
        return IR_opt_Loss, gen_recover_SNR, adver_loss_x, percept_loss

    def discr_recovery_opt(self):
        Re_inputs = self.measurement_process(self.xs_target)
        Re_recover_pre = self.integ_recovery(Re_inputs, batch_size=self.batch_size)
        Re_discr_recover = self.postprocess(Re_recover_pre, batch_size=self.batch_size)
        real_discr_value, real_percept_value = self.discriminator_for_recovery(self.xs_target)
        fake_discr_value, fake_percept_value = self.discriminator_for_recovery(Re_discr_recover)
        percept_loss = tf.reduce_sum(tf.reduce_mean(tf.square(fake_percept_value - real_percept_value),axis=0))/tf.reduce_sum(tf.reduce_mean(tf.square(real_percept_value),axis=0))
        real_loss = tf.abs(real_discr_value - self.True_Label)
        real_loss_value = 1 - tf.reduce_mean(real_discr_value[:,0] + tf.matmul(real_discr_value[:,1:], Class_Weights))
        recovery_loss_seq = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.square(self.xs_target - Re_discr_recover),axis=3),axis=2),axis=1)
        signal_seq = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.square(self.xs_target),axis=3),axis=2),axis=1)
        SNR_seq = tf.divide(signal_seq, recovery_loss_seq)
        SNR_Seq_ordered, num_seq = tf.nn.top_k(SNR_seq, self.batch_size)
        recover_ssv = structure_similar(Re_discr_recover, self.xs_target)
        Loss_mat = tf.zeros([self.batch_size, 1],dtype=tf.float32)
        Loss_mat_T = tf.zeros([self.batch_size, Discriminator_Class],dtype=tf.float32)
        for i in range(self.batch_size):
            label_class = (i//math.ceil(self.batch_size/(Discriminator_Class-1)))+1
            one_hot_value = tf.reshape(tf.one_hot([label_class],Discriminator_Class),[1,Discriminator_Class])
            empty_1 = tf.zeros([num_seq[i]+1,1],dtype=tf.float32)
            empty_2 = tf.zeros([self.batch_size-num_seq[i],1],dtype=tf.float32)
            loss_value = tf.abs(fake_discr_value[num_seq[i]] - one_hot_value)
            loss_value_vec = tf.reshape(loss_value[0,label_class]+i*0.01*loss_value[0,0], [1,1])
            loss_value_mat = tf.concat([empty_1,loss_value_vec,empty_2],0)
            Loss_mat = Loss_mat + loss_value_mat[1:-1,:]

            empty_1_t = tf.zeros([num_seq[i]+1,Discriminator_Class],dtype=tf.float32)
            empty_2_t = tf.zeros([self.batch_size-num_seq[i],Discriminator_Class],dtype=tf.float32)
            loss_value_mat_t = tf.concat([empty_1_t,loss_value,empty_2_t],0)
            Loss_mat_T = Loss_mat_T + loss_value_mat_t[1:-1,:]

            if i == 0:
                self.lossv0 = tf.matmul(tf.reshape(fake_discr_value[num_seq[i],1:],[1,Discriminator_Class-1]), Class_Weights)
                self.fkv0 = fake_discr_value[num_seq[i]]
                self.snr0 = SNR_Seq_ordered[i]
            if i == self.batch_size-1:
                self.lossv1 = tf.matmul(tf.reshape(fake_discr_value[num_seq[i],1:],[1,Discriminator_Class-1]), Class_Weights)
                self.fkv1 = fake_discr_value[num_seq[i]]
                self.snr1 = SNR_Seq_ordered[i]
                self.snr_m = tf.reduce_mean(SNR_Seq_ordered)
        fake_loss = Loss_mat
        fake_loss_sum = Loss_mat_T[:,0]
        #SNR_corelation = coefficient_correlation(SNR_seq, fake_discr_value)
        discr_ad_Loss_sum = tf.reduce_mean(real_loss[:,0] + fake_loss_sum)/2
        real_loss_coeff = 1+tf.maximum(0.0,(tf.reduce_mean(real_loss[:,0])-0.505)*tf.pow((1+real_loss_value),5))
        def fake_loss_decrease(): return (1+tf.maximum(0.0,tf.reduce_mean(fake_loss)-0.505)*tf.pow((1+tf.reduce_mean(fake_loss)),5))
        def fake_loss_increase(): return (1-tf.maximum(0.0,0.475-tf.reduce_mean(fake_loss_sum))*0)
        fake_loss_coeff = tf.cond(tf.less(tf.reduce_mean(fake_loss_sum), 0.5), fake_loss_increase, fake_loss_decrease)
        discr_SR_Loss = real_loss_coeff*real_loss_value + fake_loss_coeff*tf.reduce_mean(fake_loss) + tf.reduce_mean(tf.divide(recovery_loss_seq, signal_seq)) + percept_loss - recover_ssv
        return discr_SR_Loss, tf.reduce_mean(real_loss[:,0]), tf.reduce_mean(fake_loss_sum), discr_ad_Loss_sum, tf.reduce_mean(fake_loss)

    def train(self, Sess, SeismicSamples, ModelPath):
        # Gather variables.
        t_vars = tf.global_variables()
        self.inte_rec_vars = [var for var in t_vars if 'inte_recov_' in var.name]
        self.Dis_vars = [var for var in t_vars if 'disc_mr_' in var.name]
        self.M_opt_vars = [var for var in t_vars if 'Measurement_' in var.name]
        self.Dis_Re_vars = [var for var in t_vars if 'disc_rec_' in var.name]
        # discriminator optimizers for recovery.
        Dis_Re_optim = tf.train.AdamOptimizer(LearningRate).minimize(self.disc_re_opt_Loss, var_list=self.Dis_Re_vars)
        # recovery optimizers.
        inte_rec_optim = tf.train.AdamOptimizer(LearningRate).minimize(self.IR_opt_Loss, var_list=self.inte_rec_vars)
        # discriminator optimizers for measurement.
        Dis_optim = tf.train.AdamOptimizer(LearningRate).minimize(self.discr_opt_Loss, var_list=self.Dis_vars)
        # measurement optimizers.
        M_opt_optim = tf.train.AdamOptimizer(LearningRate).minimize(self.M_opt_Loss, var_list=self.M_opt_vars)

        batch_idxs = SeismicSamples.shape[0] // self.batch_size
        # Loading model.
        model_name = "CS-GAN.model"
        train_step = global_step
        self.counter = 0
        init = tf.global_variables_initializer()
        self.sess = Sess
        self.sess.run(init)
        print("Variables have been initialized, and the model is gong to be set up!")
        try:
            self.saver.restore(self.sess, ModelPath+model_name)
        except:
            print('there is no file to load')
            pass
        else:
            print("Model restored.")
        # Main training loop.
        epochs = Epoch_num
        self.cur_epoch = cur_epoch
        Dis_opt_Loss = 0.5
        IR_recov_SNR_sum = 0
        IR_recov_SNR_average_last = 0
        IR_recov_SNR_Max = 0
        parameter_list = []
        Dis_Re_Fake_loss = 0
        for epoch in range(self.cur_epoch, epochs):
            [_sample_num, _time_len, _sensor_num] = SeismicSamples.shape
            num_list = list(range(_sample_num))
            random.shuffle(num_list)
            SeismicSamples_random = np.zeros((_sample_num, _time_len, _sensor_num), dtype = float)
            for i in range(_sample_num):
                SeismicSamples_random[i,:,:] = SeismicSamples[num_list[i],:,:]
            for idx in range(self.counter % batch_idxs, batch_idxs):
                Mem_data = psutil.virtual_memory()
                print("Memory: %5s" % (Mem_data.percent))
                start_time = time.time()  # Time keeping.
                all_batch_images = SeismicSamples_random[idx * self.batch_size:(idx + 1) * self.batch_size,:,:]
                feed_dict = {}
                feed_dict[self.xs_target] = all_batch_images.reshape([self.batch_size]+self.image_dims)
                # Run the optimizer for the measurements and recovery.
                #Mem_data = psutil.virtual_memory()
                #print("Memory: %5s" % (Mem_data.percent))
                [M_optim_value, Meas_opt_Loss, Meas_obver_SNR, Meas_recov_SNR, Meas_ad_loss, Re_ad_loss] = self.sess.run([M_opt_optim,
                                 self.M_opt_Loss, self.M_opt_obver_SNR, self.M_opt_recov_SNR, self.M_adver_loss, self.R_adver_loss], feed_dict=feed_dict)
                if Dis_Re_Fake_loss < 0.65:
                    [Int_Recov_opt_value, IR_opt_Loss_value, IR_recov_SNR, IR_ad_loss, IR_Percept_Loss] = self.sess.run([inte_rec_optim, self.IR_opt_Loss, self.Int_Recov_SNR, self.IR_adver_Loss, self.IR_percept_Loss], feed_dict=feed_dict)
                else:
                    print("Integ_recovery passed")
                    [IR_opt_Loss_value, IR_recov_SNR, IR_ad_loss, IR_Percept_Loss] = self.sess.run([self.IR_opt_Loss, self.Int_Recov_SNR, self.IR_adver_Loss, self.IR_percept_Loss], feed_dict=feed_dict)
                # Run the optimizer for the discriminators.
                #Mem_data = psutil.virtual_memory()
                #print("Memory: %5s" % (Mem_data.percent))
                # prevent the sparse reprensentation generator from being knocked over
                if Meas_ad_loss < 0.65:
                    print("=================== measurement discriminator updated ===================")
                    [Dis_opt_value, Dis_opt_Loss, Dis_Real_loss, Dis_Fake_loss] = self.sess.run([Dis_optim, self.discr_opt_Loss, self.discr_real_loss, self.discr_fake_loss], feed_dict=feed_dict)
                else:
                    [Dis_opt_Loss, Dis_Real_loss, Dis_Fake_loss] = self.sess.run([self.discr_opt_Loss, self.discr_real_loss, self.discr_fake_loss], feed_dict=feed_dict)
                # prevent the recovery generator from being knocked over
                if IR_ad_loss < 0.65:
                    print("................... recovery discriminator updated ...................")
                    [Dis_Re_opt_value, Dis_Re_opt_Loss, Dis_Re_Real_loss, Dis_Re_Fake_loss, Dis_ad_Loss_sum, Dis_ad_reco_loss] = self.sess.run([Dis_Re_optim, self.disc_re_opt_Loss, self.disc_re_real_loss, self.disc_re_fake_loss, self.discr_Loss_sum, self.disc_recov_loss], feed_dict=feed_dict)
                else:
                    [Dis_Re_opt_Loss, Dis_Re_Real_loss, Dis_Re_Fake_loss, Dis_ad_Loss_sum, Dis_ad_reco_loss] = self.sess.run([self.disc_re_opt_Loss, self.disc_re_real_loss, self.disc_re_fake_loss, self.discr_Loss_sum, self.disc_recov_loss], feed_dict=feed_dict)
                self.counter += 1
                print("Epoch: [%2d] [%4d/%4d]； time: %4.4f; step: [%2d]"
                        % (epoch, idx, batch_idxs, time.time() - start_time, train_step))
                print("Measurement_opt_Loss: %2.4f; Measurement_obver_Loss: %2.4f; Measurement_recov_SNR: %2.4f; Measurement_adver_Loss: %2.4f; Recovery_adver_Loss: %2.4f." % (Meas_opt_Loss,
                            Meas_obver_SNR, Meas_recov_SNR, Meas_ad_loss, Re_ad_loss))
                print("IR_opt_Loss_value: %2.4f;IR_Re_SNR: %2.4f; IR_adver_loss: %2.4f; IR_Percept_Loss: %2.4f." % (IR_opt_Loss_value, -10*IR_recov_SNR, IR_ad_loss, IR_Percept_Loss))
                print("Dis_opt_Loss: %2.4f; Dis_ad_real_Loss: %2.4f; Dis_ad_fake_Loss: %2.4f." % (Dis_opt_Loss, Dis_Real_loss, Dis_Fake_loss))
                print("Dis_Re_opt_Loss: %2.4f; Dis_Re_ad_real_Loss: %2.4f; Dis_Re_ad_fake_Loss: %2.4f; Dis_ad_Loss_sum: %2.4f; Dis_ad_reco_Loss: %2.4f." % (Dis_Re_opt_Loss, Dis_Re_Real_loss, Dis_Re_Fake_loss, Dis_ad_Loss_sum, Dis_ad_reco_loss))
                print(self.sess.run(self.lossv0, feed_dict=feed_dict))
                print(self.sess.run(self.fkv0, feed_dict=feed_dict))
                print(self.sess.run(self.snr0, feed_dict=feed_dict))
                print(self.sess.run(self.lossv1, feed_dict=feed_dict))
                print(self.sess.run(self.fkv1, feed_dict=feed_dict))
                print(self.sess.run(self.snr1, feed_dict=feed_dict))
                print(self.sess.run(self.snr_m, feed_dict=feed_dict))
                # prevent the sparse reprensentation discriminator from being knocked over
                Dis_count_num = 0
                while ((Dis_opt_Loss > 0.5) and (Dis_count_num < 15)):
                    [Dis_opt_value, Dis_opt_Loss, Dis_Real_loss, Dis_Fake_loss] = self.sess.run([Dis_optim, self.discr_opt_Loss, self.discr_real_loss, self.discr_fake_loss], feed_dict=feed_dict)
                    print("Dis_opt_Loss: %2.4f; Dis_ad_real_Loss: %2.4f; Dis_ad_fake_Loss: %2.4f." % (Dis_opt_Loss, Dis_Real_loss, Dis_Fake_loss))
                    Dis_count_num = Dis_count_num + 1
                # prevent the recovery discriminator from being knocked over
                Dis_Re_count_num = 0
                while ((Dis_Re_Real_loss > 0.51 or Dis_Re_Fake_loss > 0.52) and (Dis_Re_count_num < 15)):
                    [Dis_Re_opt_value, Dis_Re_opt_Loss, Dis_Re_Real_loss, Dis_Re_Fake_loss, Dis_ad_Loss_sum, Dis_ad_reco_loss] = self.sess.run([Dis_Re_optim, self.disc_re_opt_Loss, self.disc_re_real_loss, self.disc_re_fake_loss, self.discr_Loss_sum, self.disc_recov_loss], feed_dict=feed_dict)
                    print("Dis_Re_opt_Loss: %2.4f; Dis_Re_ad_real_Loss: %2.4f; Dis_Re_ad_fake_Loss: %2.4f; Dis_ad_Loss_sum: %2.4f; Dis_ad_reco_loss: %2.4f." % (Dis_Re_opt_Loss, Dis_Re_Real_loss, Dis_Re_Fake_loss, Dis_ad_Loss_sum, Dis_ad_reco_loss))
                    Dis_Re_count_num = Dis_Re_count_num + 1
                parameter_line = []
                parameter_line.append(-10*IR_recov_SNR)
                parameter_line.append(IR_ad_loss)
                parameter_line.append(Dis_Re_Real_loss)
                parameter_line.append(Dis_Re_Fake_loss)
                parameter_list.append(parameter_line)
                # Save model.
                if self.counter % max(1, int(batch_idxs // Save_Times)) == 0:
                    if IR_recov_SNR_Max > IR_recov_SNR:
                        self.saver.save(self.sess, os.path.join(ModelPath, model_name))
                        print("model has just been saved")
                        IR_recov_SNR_Max = IR_recov_SNR
                    #saving evaluation parameters
                    recording_parameters = parameter_list[self.counter-int(batch_idxs // Save_Times):]
                    with open(ModelPath+'evaluation_parameters.csv', 'a+', newline='') as parameters_file:
                        wr = csv.writer(parameters_file, dialect='excel', quoting=csv.QUOTE_NONNUMERIC)
                        for i in range(int(batch_idxs // Save_Times)):
                            wr.writerow(recording_parameters[i])
                IR_recov_SNR_sum = IR_recov_SNR_sum + IR_recov_SNR
                if self.counter % max(1, int(batch_idxs)) == 0:
                    IR_recov_SNR_average = IR_recov_SNR_sum / batch_idxs
                    if (IR_recov_SNR_average_last - IR_recov_SNR_average) < 0.05:
                        print(" !!!!!! there is a bottleneck for learning !!!!!! ")
                    IR_recov_SNR_average_last = IR_recov_SNR_average
                    print("the average SNR is %2.4f." % (IR_recov_SNR_average_last))
                    IR_recov_SNR_sum = 0
                train_step = train_step + 1

def main():
    model_path = os.getcwd()+"/model/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    data_path = os.getcwd()+"/test_real_data/"
    seismic_samples = load_seismic_sample(data_path)
    [sample_num, time_len, sensor_num] = seismic_samples.shape
    CS_Number = int(sensor_num/CS_Number_Ratio)
    BatchSize = sample_num
    print(seismic_samples.shape)
    csgan = CSGAN(time_len, sensor_num, BatchSize, CS_Number, Recov_Channel, DisMeas_Channel, DisReco_Channel)
    print("The model has been initialized!")
    with tf.Session() as sess:
        csgan.train(sess, seismic_samples, model_path)


if __name__ == '__main__':
    main()
