import tensorflow as tf


def logistic_full_mrf_loss(xw, ica_a_flat):
    ica_a_pos = tf.nn.softplus(ica_a_flat)
    xw_abs = tf.abs(xw)
    log_sum_exp = tf.log(1 + tf.exp(-2 * xw_abs)) + xw_abs
    neg_g_wx = (tf.log(0.5) + log_sum_exp) * ica_a_pos
    neg_log_p_patches = tf.reduce_sum(neg_g_wx, axis=1)
    return tf.reduce_mean(neg_log_p_patches, name='loss')


def logistic_full_score_matching_loss(x_tsr, ica_w, ica_a):
    ica_a_pos = tf.nn.softplus(ica_a)
    const_t = x_tsr.get_shape()[0].value
    xw_mat = tf.matmul(x_tsr, ica_w)
    g_mat = -tf.tanh(xw_mat)
    gp_mat = -4.0 / tf.square(tf.exp(xw_mat) + tf.exp(-xw_mat))  # d/dx tanh(x) = 4 / (exp(x) + exp(-x))^2
    gp_vec = tf.reduce_sum(gp_mat, axis=0) / const_t
    gg_mat = tf.matmul(g_mat, g_mat, transpose_a=True) / const_t
    aa_mat = tf.matmul(ica_a_pos, ica_a_pos, transpose_b=True)
    ww_mat = tf.matmul(ica_w, ica_w, transpose_a=True)
    w_norm = tf.diag_part(ww_mat, name='w_norm')

    term_1 = tf.reduce_sum(ica_a_pos * w_norm * gp_vec, name='t1')
    term_2 = 0.5 * tf.reduce_sum(aa_mat * ww_mat * gg_mat, name='t2')
    return term_1 + term_2, term_1, term_2


def logistic_channelwise_mrf_loss(xw, ica_a_flat):
    ica_a_pos = tf.nn.softplus(ica_a_flat)
    xw_abs = tf.abs(xw)
    log_sum_exp = tf.log(1 + tf.exp(-2 * xw_abs)) + xw_abs
    neg_g_wx = (tf.log(0.5) + log_sum_exp) * tf.stack([ica_a_pos] * xw.get_shape()[1].value, axis=1)
    neg_log_p_patches = tf.reduce_sum(neg_g_wx, axis=1)
    return tf.reduce_mean(neg_log_p_patches, name='loss')


def logistic_channelwise_score_matching_loss(x_tsr, ica_w, ica_a):
    ica_a_pos = tf.nn.softplus(ica_a)
    const_t = x_tsr.get_shape()[0].value
    xw_mat = tf.matmul(x_tsr, ica_w)
    g_mat = -tf.tanh(xw_mat)
    gp_mat = -4.0 / tf.square(tf.exp(xw_mat) + tf.exp(-xw_mat))  # d/dx tanh(x) = 4 / (exp(x) + exp(-x))^2
    gp_vec = tf.reduce_sum(gp_mat, axis=1) / const_t
    gg_mat = tf.matmul(g_mat, g_mat, transpose_a=True) / const_t
    aa_mat = tf.matmul(ica_a_pos, ica_a_pos, transpose_b=True)
    ww_mat = tf.matmul(ica_w, ica_w, transpose_a=True)

    ww_list = tf.split(ww_mat, num_or_size_splits=ww_mat.get_shape()[0].value, axis=0)
    w_norm_list = [tf.diag_part(tf.squeeze(k)) for k in ww_list]
    w_norm = tf.stack(w_norm_list, axis=0, name='w_norm')

    term_1 = tf.reduce_sum(tf.squeeze(ica_a_pos) * w_norm * gp_vec, name='t1')
    term_2 = 0.5 * tf.reduce_sum(aa_mat * ww_mat * gg_mat, name='t2')
    return term_1 + term_2, term_1, term_2


def student_full_mrf_loss(xw, ica_a_flat):
    ica_a_pos = tf.nn.softplus(ica_a_flat)
    neg_g_wx = tf.log(1.0 + 0.5 * tf.square(xw)) * ica_a_pos
    neg_log_p_patches = tf.reduce_sum(neg_g_wx, axis=1)
    return tf.reduce_mean(neg_log_p_patches, name='loss')


def student_full_score_matching_loss(x_tsr, ica_w, ica_a):
    ica_a_pos = tf.nn.softplus(ica_a)
    const_t = x_tsr.get_shape()[0].value
    xw_mat = tf.matmul(x_tsr, ica_w)
    xw_square_mat = tf.square(xw_mat)
    g_mat = - 2 * xw_mat / (xw_square_mat + 2)
    gp_mat = 2.0 * (xw_square_mat - 2) / tf.square(xw_square_mat + 2)  # d/dx tanh(x) = 4 / (exp(x) + exp(-x))^2
    gp_vec = tf.reduce_sum(gp_mat, axis=0) / const_t
    gg_mat = tf.matmul(g_mat, g_mat, transpose_a=True) / const_t
    aa_mat = tf.matmul(ica_a_pos, ica_a_pos, transpose_b=True)
    ww_mat = tf.matmul(ica_w, ica_w, transpose_a=True)
    w_norm = tf.diag_part(ww_mat, name='w_norm')

    term_1 = tf.reduce_sum(ica_a_pos * w_norm * gp_vec, name='t1')
    term_2 = 0.5 * tf.reduce_sum(aa_mat * ww_mat * gg_mat, name='t2')
    return term_1 + term_2, term_1, term_2


def student_channelwise_mrf_loss(xw, ica_a_flat):
    ica_a_pos = tf.nn.softplus(ica_a_flat)
    neg_g_wx = tf.log(1.0 + 0.5 * tf.square(xw)) * tf.stack([ica_a_pos] * xw.get_shape()[1].value, axis=1)
    neg_log_p_patches = tf.reduce_sum(neg_g_wx, axis=1)
    return tf.reduce_mean(neg_log_p_patches, name='loss')


def student_channelwise_score_matching_loss(x_tsr, ica_w, ica_a):
    ica_a_pos = tf.nn.softplus(ica_a)
    const_t = x_tsr.get_shape()[0].value
    xw_mat = tf.matmul(x_tsr, ica_w)
    xw_square_mat = tf.square(xw_mat)
    g_mat = - 2 * xw_mat / (xw_square_mat + 2)
    gp_mat = 2.0 * (xw_square_mat - 2) / tf.square(xw_square_mat + 2)  # d/dx tanh(x) = 4 / (exp(x) + exp(-x))^2
    gp_vec = tf.reduce_sum(gp_mat, axis=1) / const_t

    gg_mat = tf.matmul(g_mat, g_mat, transpose_a=True) / const_t
    aa_mat = tf.matmul(ica_a_pos, ica_a_pos, transpose_b=True)
    ww_mat = tf.matmul(ica_w, ica_w, transpose_a=True)

    ww_list = tf.split(ww_mat, num_or_size_splits=ww_mat.get_shape()[0].value, axis=0)
    w_norm_list = [tf.diag_part(tf.squeeze(k)) for k in ww_list]
    w_norm = tf.stack(w_norm_list, axis=0, name='w_norm')

    term_1 = tf.reduce_sum(tf.squeeze(ica_a_pos) * w_norm * gp_vec, name='t1')
    term_2 = 0.5 * tf.reduce_sum(aa_mat * ww_mat * gg_mat, name='t2')
    return term_1 + term_2, term_1, term_2
