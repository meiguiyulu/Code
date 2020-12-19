import numpy as np
import os
import csv
import tensorflow as tf
import math
import sys

from csgan_utils import *

CS_Number_Ratio = 16

Recov_Channel = 8         # recovery's convolutional channel dimension
DisMeas_Channel = 8       # Measurement Discriminator's convolutional channel dimension
DisReco_Channel = 8       # Recovery Discriminator's convolutional channel dimension

MinConInd = 0.01

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
        self.IR_opt_Loss, self.Int_Recov_SNR, self.IR_adver_Loss, self.IR_recover_data = self.integ_recovery_opt()
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

    def integ_recovery_opt(self):
        Re_inputs = self.measurement_process(self.xs_target)
        Re_recover_pre = self.integ_recovery(Re_inputs, batch_size=self.batch_size)
        Re_gen_recover = self.postprocess(Re_recover_pre, batch_size=self.batch_size)
        Re_recovery_loss_seq = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.square(self.xs_target - Re_gen_recover),axis=3),axis=2),axis=1)
        Re_signal_seq = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.square(self.xs_target),axis=3),axis=2),axis=1)
        gen_recover_SNR_n = tf.reduce_mean(tf.divide(Re_recovery_loss_seq,Re_signal_seq))
        gen_recover_SNR = tf_log_10(gen_recover_SNR_n)
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
        def adver_loss_reduce(): return 1+tf.maximum(0.0,(tf.reduce_mean(adver_loss_x)-0.5)*tf.pow((1+tf.reduce_mean(adver_loss_x)),2))
        def adver_loss_enlarge(): return tf.maximum(0.0,1-tf.maximum(0.0,0.5-tf.reduce_mean(adver_loss_x))*0)
        adver_loss_coeff = tf.cond(tf.less(tf.reduce_mean(adver_loss_x), 0.5), adver_loss_enlarge, adver_loss_reduce)
        IR_opt_Loss = gen_recover_SNR + diff_ssv + adver_loss_coeff*adver_loss_Re + percept_loss
        return IR_opt_Loss, gen_recover_SNR, adver_loss_x, Re_gen_recover

    def test(self, Sess, SeismicSamples, ModelPath, save_path):
        # Loading model.
        model_name = "CS-GAN.model"
        init = tf.global_variables_initializer()
        self.sess = Sess
        self.sess.run(init)
        print("Variables have been initialized, and the model is gong to be set up!")
        try:
            self.saver.restore(self.sess, ModelPath+model_name)
        except:
            print('there is no proper model to be loaded, and the files in this directory are shown as follow:')
            object_dir = os.listdir(ModelPath)
            for file in object_dir:
                print(file)
            sys.exit(0)
        else:
            print("Model restored.")
        # Main test process.
        batch_idxs = SeismicSamples.shape[0] // self.batch_size
        for idx in range(batch_idxs):
            all_batch_images = SeismicSamples[idx * self.batch_size:(idx + 1) * self.batch_size,:,:]
            feed_dict = {}
            feed_dict[self.xs_target] = all_batch_images.reshape([self.batch_size]+self.image_dims)
            # execute the recovery.
            [IR_Loss, IR_recov_SNR, IR_ad_Loss, recover_data] = self.sess.run([self.IR_opt_Loss, self.Int_Recov_SNR, self.IR_adver_Loss, self.IR_recover_data], feed_dict=feed_dict)
            print("IR_opt_Loss: %2.4f; IR_adver_Loss: %2.4f; IR_Re_SNR: %2.4f." % (IR_Loss, IR_ad_Loss, -10*IR_recov_SNR))
            for i in range(self.batch_size):
                with open(save_path+'recover_data_'+str(idx*self.batch_size+i)+'.csv', 'w', newline='') as save_file:
                    wr = csv.writer(save_file, dialect='excel', quoting=csv.QUOTE_NONNUMERIC)
                    for j in range(self.input_height):
                        wr.writerow(np.array(recover_data[i,j,:,0]))

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
    result_path = os.getcwd()+"/TestResult/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    csgan = CSGAN(time_len, sensor_num, BatchSize, CS_Number, Recov_Channel, DisMeas_Channel, DisReco_Channel)
    print("The model has been initialized!")
    with tf.Session() as sess:
        csgan.test(sess, seismic_samples, model_path, result_path)


if __name__ == '__main__':
    main()
