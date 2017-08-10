import tensorflow as tf
import numpy as np
from modules.ica_prior import ICAPrior
from utils.temp_utils import flattening_filter, patch_batch_gen, plot_img_mats
from utils.preprocessing import preprocess_tensor
import time
import os


class FoEPrior(ICAPrior):

    def __init__(self, tensor_names, weighting, classifier, filter_dims, input_scaling, n_components, n_channels,
                 n_features_white,
                 trainable=False, name='FoEPrior', load_name='FoEPrior', dir_name='foe_prior',
                 mean_mode='lf', sdev_mode='none'):
        super().__init__(tensor_names, weighting, classifier, filter_dims, input_scaling, n_components, n_channels,
                         n_features_white, trainable=trainable, name=name, load_name=load_name, dir_name=dir_name,
                         mean_mode=mean_mode, sdev_mode=sdev_mode)

    def build(self, scope_suffix=''):
        with tf.variable_scope(self.name):
            tensor = self.get_tensors()
            scaled_tensor = tensor * self.input_scaling

            # filter_mat = flattening_filter((self.filter_dims[0], self.filter_dims[1], dims[3]))
            filter_mat = flattening_filter((self.filter_dims[0], self.filter_dims[1], 1))

            flat_filter = tf.constant(filter_mat, dtype=tf.float32)
            x_pad = ((self.filter_dims[0] - 1) // 2, int(np.ceil((self.filter_dims[0] - 1) / 2)))
            y_pad = ((self.filter_dims[1] - 1) // 2, int(np.ceil((self.filter_dims[1] - 1) / 2)))
            conv_input = tf.pad(scaled_tensor, paddings=[(0, 0), x_pad, y_pad, (0, 0)], mode='REFLECT')
            # flat_patches = tf.nn.conv2d(conv_input, flat_filter, strides=[1, 1, 1, 1], padding='VALID')

            feat_map_list = tf.split(conv_input, num_or_size_splits=conv_input.get_shape()[3].value, axis=3)
            flat_patches_list = [tf.nn.conv2d(k, flat_filter, strides=[1, 1, 1, 1], padding='VALID')
                                 for k in feat_map_list]  # first, filter per channel
            flat_patches = tf.stack(flat_patches_list, axis=4)
            fps = [k.value for k in flat_patches.get_shape()]  # shape=[bs, h, w, n_fpc, n_c]
            n_patches = fps[1] * fps[2]
            n_features_raw = fps[3] * fps[4]
            flat_patches = tf.reshape(flat_patches, shape=[fps[0], n_patches, fps[3], fps[4]])  # flatten h,w
            assert fps[0] == 1  # commit to singular batch size
            flat_patches = tf.squeeze(flat_patches)
            print(1, flat_patches.get_shape())

            normed_patches = preprocess_tensor(flat_patches, self.mean_mode, self.sdev_mode)
            normed_patches = tf.reshape(normed_patches, shape=[n_patches, n_features_raw])

            # scaled_patches = flat_patches  # * self.input_scaling
            # centered_patches = flat_patches - tf.stack([tf.reduce_mean(scaled_patches, axis=3)] * filter_mat.shape[3],
            #                                            axis=3)
            # n_features_raw = filter_mat.shape[3]
            # centered_patches = tf.reshape(centered_patches, shape=[-1, n_features_raw])

            whitening_tensor = tf.get_variable('whiten_mat', shape=[self.n_features_white, n_features_raw],
                                               dtype=tf.float32, trainable=self.trainable)
            ica_a = tf.get_variable('ica_a', shape=[self.n_components, 1], trainable=self.trainable, dtype=tf.float32)
            ica_w = tf.get_variable('ica_w', shape=[self.n_features_white, self.n_components],
                                    trainable=self.trainable, dtype=tf.float32)
            ica_a_squeezed = tf.squeeze(ica_a)

            whitened_mixing = tf.matmul(whitening_tensor, ica_w, transpose_a=True)
            xw = tf.matmul(normed_patches, whitened_mixing)
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
