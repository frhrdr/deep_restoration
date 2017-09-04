import tensorflow as tf
from modules.foe_full_prior import FoEFullPrior
from utils.patch_prior_losses import student_full_mrf_loss, student_full_score_matching_loss

class FoEPrior(FoEFullPrior):

    def __init__(self, tensor_names, weighting, classifier, filter_dims, input_scaling, n_components, n_channels,
                 n_features_white,
                 trainable=False, name='FoEPrior', load_name='FoEPrior', dir_name='foe_prior',
                 mean_mode='lf', sdev_mode='none', load_tensor_names=None):
        super().__init__(tensor_names, weighting, classifier, filter_dims, input_scaling, n_components, n_channels,
                         n_features_white, trainable=trainable, name=name, load_name=load_name, dir_name=dir_name,
                         mean_mode=mean_mode, sdev_mode=sdev_mode, load_tensor_names=load_tensor_names)

    @staticmethod
    def score_matching_loss(x_mat, w_mat, alpha):
        return student_full_score_matching_loss(x_mat, w_mat, alpha)
        # const_T = x_mat.get_shape()[0].value
        # xw_mat = tf.matmul(x_mat, w_mat)
        # xw_square_mat = tf.square(xw_mat)
        # g_mat = - 2 * xw_mat / (xw_square_mat + 2)
        # gp_mat = 2.0 * (xw_square_mat - 2) / tf.square(xw_square_mat + 2)  # d/dx tanh(x) = 4 / (exp(x) + exp(-x))^2
        # gp_vec = tf.reduce_sum(gp_mat, axis=0) / const_T
        # gg_mat = tf.matmul(g_mat, g_mat, transpose_a=True) / const_T
        # aa_mat = tf.matmul(alpha, alpha, transpose_b=True)
        # ww_mat = tf.matmul(w_mat, w_mat, transpose_a=True)
        # w_norm = tf.diag_part(ww_mat, name='w_norm')
        #
        # term_1 = tf.reduce_sum(alpha * w_norm * gp_vec, name='t1')
        # term_2 = 0.5 * tf.reduce_sum(aa_mat * ww_mat * gg_mat, name='t2')
        # return term_1 + term_2, term_1, term_2

    @staticmethod
    def mrf_loss(xw, ica_a_squeezed):
        return student_full_mrf_loss(xw, ica_a_squeezed)
        # neg_g_wx = tf.log(1.0 + 0.5 * tf.square(xw)) * ica_a_squeezed
        # neg_log_p_patches = tf.reduce_sum(neg_g_wx, axis=1)
        # return tf.reduce_mean(neg_log_p_patches, name='loss')