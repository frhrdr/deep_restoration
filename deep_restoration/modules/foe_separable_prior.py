import tensorflow as tf
import numpy as np
import time
import os
from modules.loss_modules import LearnedPriorLoss
from utils.temp_utils import flattening_filter, patch_batch_gen, plot_img_mats, get_optimizer
from utils.preprocessing import preprocess_patch_tensor, make_data_dir, preprocess_featmap_tensor
from modules.foe_full_prior import FoEFullPrior


class FoESeparablePrior(FoEFullPrior):

    def __init__(self, tensor_names, weighting, classifier, filter_dims, input_scaling, n_components, n_channels,
                 n_features_white, dist='logistic', mean_mode='gc', sdev_mode='gc',
                 trainable=False, name=None, load_name=None, dir_name=None, load_tensor_names=None):

        super().__init__(tensor_names, weighting, classifier, filter_dims, input_scaling, n_components, n_channels,
                         n_features_white, dist=dist, mean_mode=mean_mode, sdev_mode=sdev_mode, trainable=trainable,
                         name=name, load_name=load_name, dir_name=dir_name, load_tensor_names=load_tensor_names)


    def build(self, scope_suffix=''):

        with tf.variable_scope(self.name):

            n_features_per_channel = self.filter_dims[0] * self.filter_dims[1]
            n_features_raw = n_features_per_channel * self.n_channels
            whitening_tensor = tf.get_variable('whiten_mat', shape=[self.n_features_white, n_features_raw],
                                               dtype=tf.float32, trainable=False)

            ica_a = tf.get_variable('ica_a', shape=[self.n_components, 1], trainable=self.trainable, dtype=tf.float32)
            ica_a_squeezed = tf.squeeze(ica_a)

            depth_filter = tf.get_variable('depth_filter', shape=[self.n_components, self.n_features_white])

            point_filter = tf.get_variable('point_filter', shape=[])
            ica_w = None

            whitened_mixing = tf.matmul(whitening_tensor, ica_w, transpose_a=True)

            if self.mean_mode in ('gc', 'gf') and self.sdev_mode in ('gc', 'gf'):
                normed_featmap = self.norm_feat_map_directly()  # shape [1, h, w, n_channels]

                whitened_mixing = tf.reshape(whitened_mixing, shape=[self.n_channels, self.filter_dims[0],
                                                                     self.filter_dims[1], self.n_components])
                whitened_mixing = tf.transpose(whitened_mixing, perm=[1, 2, 0, 3])
                print(normed_featmap.get_shape())
                print(whitened_mixing.get_shape())
                xw = tf.nn.conv2d(normed_featmap, whitened_mixing, strides=[1, 1, 1, 1], padding='VALID')
                xw = tf.reshape(xw, shape=[-1, self.n_components])

            else:
                normed_patches = self.shape_and_norm_tensor()
                n_patches = normed_patches.get_shape()[0].value
                normed_patches = tf.reshape(normed_patches, shape=[n_patches, n_features_raw])
                xw = tf.matmul(normed_patches, whitened_mixing)

            self.loss = self.mrf_loss(xw, ica_a_squeezed)
            self.var_list.extend([ica_a, ica_w, whitening_tensor])