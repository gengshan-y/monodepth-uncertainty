# e Copyright UCL Business plc 2017. Patent Pending. All rights reserved. 
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence, 
# please contact info@uclb.com

"""Fully convolutional model for monocular depth estimation
    by Clement Godard, Oisin Mac Aodha and Gabriel J. Brostow
    http://visual.cs.ucl.ac.uk/pubs/monoDepth/
"""

from __future__ import absolute_import, division, print_function
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import math_ops
from utils import resnet_v1
import configparser
import pdb


def get_stats(prob, n3, my3bins):
    max_idx = tf.argmax(prob,-1)

    most_likely = tf.tensordot(tf.one_hot(max_idx,n3),my3bins,[-1,0])
    most_likely = tf.exp(tf.expand_dims(most_likely,-1))

    expectation = tf.tensordot(prob,my3bins,[-1,0])
    expectation = tf.exp(tf.expand_dims(expectation,-1))

    entropy = -tf.reduce_sum(prob*tf.log(prob),-1,keepdims=True)
    conf = tf.reduce_max(prob,-1,keepdims=True)

    return most_likely, expectation, entropy, conf



class MonodepthModel(object):
    """monodepth model"""

    def __init__(self, params, mode, left, shape, reuse_variables=None):
        config = configparser.RawConfigParser()
        config.read(params.config_path)
        self.n3 = int(config.get('model', 'n3'))
        self.y3bins = np.linspace(np.log(0.5), np.log(80.), self.n3).astype(np.float32)
        self.shape =  shape
        self.my3bins = (self.y3bins + tf.concat((self.y3bins[0:1],self.y3bins[:-1]),axis=0)) / 2

        self.params = params
        self.mode = mode
        self.left = left
        if self.mode == 'test':
            self.is_training = False
        else:
            self.is_training = True
        self.reuse_variables = reuse_variables

        self.build_model()


    def conv(self, x, num_out_layers, kernel_size, stride, \
             activation_fn=tf.nn.elu, is_batchnorm = True, scope=None):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        if is_batchnorm:
            return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID',\
               activation_fn=activation_fn, normalizer_fn = slim.batch_norm, scope=scope)
        else:
            return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID',\
               activation_fn=activation_fn, scope=scope)

    def conv_block(self, x, num_out_layers, kernel_size):
        conv1 = self.conv(x,     num_out_layers, kernel_size, 1)
        conv2 = self.conv(conv1, num_out_layers, kernel_size, 2)
        return conv2


    def upsample_nn(self, x, ratio):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])


    def upconv(self, x, num_out_layers, kernel_size, scale):
        upsample = self.upsample_nn(x, scale)
        conv = self.conv(upsample, num_out_layers, kernel_size, 1)
        return conv


    def dropout(self,x):
        if self.mode == 'train':
            return tf.nn.dropout(x,0.5)
        else:
            return tf.nn.dropout(x,1.0)

    def build_resnet50(self):
        #set convenience functions
        conv   = self.conv
        upconv = self.upconv

        with tf.variable_scope('encoder'):
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                conv5, end_points = resnet_v1.resnet_v1_50(self.model_input, is_training=self.is_training, global_pool=False, dropout=self.params.dropout)
                conv4 = end_points['model/encoder/resnet_v1_50/block3']
                conv3 = end_points['model/encoder/resnet_v1_50/block2']
                conv2 = end_points['model/encoder/resnet_v1_50/block1']
                conv1 = end_points['model/encoder/resnet_v1_50/conv1']
                pool1 = slim.max_pool2d(conv1,3)

                # global encoder
                g_shape = [self.shape[0]//32,self.shape[1]//32]
                global_pool1 = tf.reduce_mean(conv5,axis=[1,2],keepdims=True)
                global_pool2 = tf.nn.avg_pool(conv5, (1,g_shape[0]//2,g_shape[1]//2,1),(1,g_shape[0]//2,g_shape[1]//2,1),'VALID')
                global_pool3 = tf.nn.avg_pool(conv5, (1,g_shape[0]//4,g_shape[1]//4,1),(1,g_shape[0]//4,g_shape[1]//4,1),'VALID')
                global_pool4 = tf.nn.avg_pool(conv5, (1,g_shape[0]//8,g_shape[1]//8,1),(1,g_shape[0]//8,g_shape[1]//8,1),'VALID')

                global_conv1 = self.conv(global_pool1, 512, 1,1)
                global_conv2 = self.conv(global_pool2, 512, 1,1)
                global_conv3 = self.conv(global_pool3, 512, 1,1)
                global_conv4 = self.conv(global_pool4, 512, 1,1)
                # bilinear upsampling
                conv5spp = tf.concat((conv5,\
                                   tf.image.resize_images(global_conv1,(g_shape[0],g_shape[1])),\
                                   tf.image.resize_images(global_conv2,(g_shape[0],g_shape[1])),\
                                   tf.image.resize_images(global_conv3,(g_shape[0],g_shape[1])),\
                                   tf.image.resize_images(global_conv4,(g_shape[0],g_shape[1]))),-1)
                conv5spp = conv(conv5spp,   2048, 1, 1) #H/32

        # DECODING
        with tf.variable_scope('decoder'):
            concat6 = tf.concat([conv5spp, conv4], 3)
            iconv6  = conv(concat6,   512, 3, 1)
            if self.params.dropout:
                iconv6 = tf.nn.dropout(iconv6,0.5)

            upconv5 = upconv(iconv6, 256, 3, 2) #H/16
            concat5 = tf.concat([upconv5, conv3], 3)
            iconv5  = conv(concat5,   256, 3, 1)
            if self.params.dropout:
                iconv5 = tf.nn.dropout(iconv5,0.5)

            upconv4 = upconv(iconv5,  128, 3, 2) #h/8
            concat4 = tf.concat([upconv4, conv2], 3)
            iconv4  = conv(concat4,   128, 3, 1)
            if self.params.dropout:
                iconv4 = tf.nn.dropout(iconv4,0.5)

            upconv3 = upconv(iconv4,   128, 3, 2) #h/4
            concat3 = tf.concat([upconv3, pool1], 3)
            iconv3  = conv(concat3,    128, 3, 1)

            upconv2 = upconv(iconv3,   128, 3, 2) #h/2
            concat2 = tf.concat([upconv2, conv1], 3)
            iconv2  = conv(concat2,    128, 3, 1)

            prediction_layer_name = None
            self.logits = conv(iconv2, self.n3, 3, 1,\
                          None,is_batchnorm=False,scope=prediction_layer_name)
            self.logits = tf.image.resize_images(self.logits, [self.shape[0],self.shape[1]])

            # inference            
            self.prob = tf.nn.softmax(self.logits,-1)
            self.resp = -tf.reduce_max(self.logits,-1,keepdims=True)
            self.most_likely, self.expectation, self.entropy, self.conf = get_stats(self.prob,self.n3,self.my3bins)


    def build_model(self):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            with tf.variable_scope('model', reuse=self.reuse_variables):
                self.model_input = self.left
                
                #build model
                self.build_resnet50()

