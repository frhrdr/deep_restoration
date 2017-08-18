import tensorflow as tf
from modules.ica_prior import ICAPrior


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
            normed_patches = self.shape_and_norm_tensor()
            n_patches, n_channels, n_feats_per_channel = [k.value for k in normed_patches.get_shape()]
            n_features_raw = n_feats_per_channel * n_channels

            normed_patches = tf.reshape(normed_patches, shape=[n_patches, n_features_raw])

            whitening_tensor = tf.get_variable('whiten_mat', shape=[self.n_features_white, n_features_raw],
                                               dtype=tf.float32, trainable=False)

            ica_a = tf.get_variable('ica_a', shape=[self.n_components, 1], trainable=self.trainable, dtype=tf.float32)
            ica_w = tf.get_variable('ica_w', shape=[self.n_features_white, self.n_components],
                                    trainable=self.trainable, dtype=tf.float32)
            ica_a_squeezed = tf.squeeze(ica_a)
            whitened_mixing = tf.matmul(whitening_tensor, ica_w, transpose_a=True)

            xw = tf.matmul(normed_patches, whitened_mixing)
            neg_g_wx = tf.log(1.0 + 0.5 * tf.square(xw)) * ica_a_squeezed
            neg_log_p_patches = tf.reduce_sum(neg_g_wx, axis=1)
            self.loss = tf.reduce_mean(neg_log_p_patches, name='loss')

            self.var_list.extend([ica_a, ica_w, whitening_tensor])

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
