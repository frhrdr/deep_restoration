import tensorflow as tf
import numpy as np
from modules.ica_prior import ICAPrior
from utils.temp_utils import flattening_filter, patch_batch_gen, plot_img_mats
import time
import os


class FoEPrior(ICAPrior):

    def __init__(self, tensor_names, weighting, name, classifier, trainable,
                 filter_dims, input_scaling, n_components,  n_channels, n_features_white,
                 load_name='FoEPrior', dir_name='foe_prior'):
        super().__init__(tensor_names, weighting, name, classifier, trainable,
                         filter_dims, input_scaling, n_components,  n_channels,
                         n_features_white, load_name=load_name, dir_name=dir_name)

    def build(self, scope_suffix=''):
        with tf.variable_scope(self.name):
            tensor = self.get_tensors()
            dims = [s.value for s in tensor.get_shape()]
            assert len(dims) == 4
            filter_mat = flattening_filter((self.filter_dims[0], self.filter_dims[1], dims[3]))
            flat_filter = tf.constant(filter_mat, dtype=tf.float32)
            x_pad = ((self.filter_dims[0] - 1) // 2, int(np.ceil((self.filter_dims[0] - 1) / 2)))
            y_pad = ((self.filter_dims[1] - 1) // 2, int(np.ceil((self.filter_dims[1] - 1) / 2)))
            print(x_pad, y_pad)
            conv_input = tf.pad(tensor, paddings=[(0, 0), x_pad, y_pad, (0, 0)], mode='REFLECT')
            flat_patches = tf.nn.conv2d(conv_input, flat_filter, strides=[1, 1, 1, 1], padding='VALID')
            scaled_patches = flat_patches  # * self.input_scaling
            centered_patches = flat_patches - tf.stack([tf.reduce_mean(scaled_patches, axis=3)] * filter_mat.shape[3],
                                                       axis=3)

            n_features_raw = filter_mat.shape[3]

            whitening_tensor = tf.get_variable('whiten_mat', shape=[self.n_features_white, n_features_raw],
                                               dtype=tf.float32, trainable=self.trainable)

            centered_patches = tf.reshape(centered_patches, shape=[-1, n_features_raw])

            ica_a = tf.get_variable('ica_a', shape=[self.n_components, 1], trainable=self.trainable, dtype=tf.float32)
            ica_w = tf.get_variable('ica_w', shape=[self.n_features_white, self.n_components],
                                    trainable=self.trainable, dtype=tf.float32)
            ica_a_squeezed = tf.squeeze(ica_a)

            whitened_mixing = tf.matmul(whitening_tensor, ica_w, transpose_a=True)
            xw = tf.matmul(centered_patches, whitened_mixing)
            neg_g_wx = tf.log(1.0 + 0.5 * tf.square(xw)) * ica_a_squeezed
            neg_log_p_patches = tf.reduce_sum(neg_g_wx, axis=1)
            naive_mean = tf.reduce_mean(neg_log_p_patches, name='loss')
            self.loss = naive_mean

            self.var_list = [ica_a, ica_w, whitening_tensor]

    @staticmethod
    def score_matching_loss(x_mat, w_mat, alpha):
        const_T = x_mat.get_shape()[0].value
        xw_mat = tf.matmul(x_mat, w_mat)
        xw_square_mat = tf.square(xw_mat)
        g_mat = - 2 * xw_mat / (xw_square_mat + 2)
        gp_mat = 2.0 * (xw_square_mat - 2) / tf.square(xw_square_mat + 2)  # d/dx tanh(x) = 4 / (exp(x) + exp(-x))^2
        gp_vec = tf.reduce_sum(gp_mat, axis=0) / const_T
        gg_mat = tf.matmul(g_mat, g_mat, transpose_a=True) / const_T
        aa_mat = tf.matmul(alpha, alpha, transpose_b=True)
        ww_mat = tf.matmul(w_mat, w_mat, transpose_a=True)
        w_norm = tf.diag_part(ww_mat, name='w_norm')

        term_1 = tf.reduce_sum(alpha * w_norm * gp_vec, name='t1')
        term_2 = 0.5 * tf.reduce_sum(aa_mat * ww_mat * gg_mat, name='t2')
        return term_1 + term_2, term_1, term_2
