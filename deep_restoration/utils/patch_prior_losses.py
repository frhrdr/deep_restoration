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

    term_1 = tf.reduce_sum(ica_a_pos * gp_vec, name='t1')
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
    const_t = x_tsr.get_shape()[1].value
    xw_mat = tf.matmul(x_tsr, ica_w)
    g_mat = -tf.tanh(xw_mat)
    gp_mat = -4.0 / tf.square(tf.exp(xw_mat) + tf.exp(-xw_mat))  # d/dx tanh(x) = 4 / (exp(x) + exp(-x))^2
    gp_vec = tf.reduce_sum(gp_mat, axis=1) / const_t
    gg_mat = tf.matmul(g_mat, g_mat, transpose_a=True) / const_t
    aa_mat = tf.matmul(ica_a_pos, ica_a_pos, transpose_b=True)
    ww_mat = tf.matmul(ica_w, ica_w, transpose_a=True)

    term_1 = tf.reduce_sum(tf.squeeze(ica_a_pos) * gp_vec, name='t1')
    term_2 = 0.5 * tf.reduce_sum(aa_mat * ww_mat * gg_mat, name='t2')
    return term_1 + term_2, term_1, term_2


def logistic_separable_score_matching_map_fun(loss_acc, index, x_tsr, depth_filter, point_filter, ica_a):
    w_k = get_single_separable_filter(depth_filter, point_filter, index)
    a_k = tf.slice(ica_a, [index], [1])
    xw_mat = tf.reduce_sum(x_tsr * w_k, axis=(1, 2))  # needs reviewing
    g_patch = tf.reduce_sum(-tf.tanh(xw_mat))
    term_1 = a_k * g_patch

    term_acc_init = (0.,)
    term_2 = tf.foldl(lambda x, y: logistic_separable_nested_sm_map_fun(x, y, x_tsr, a_k, w_k, depth_filter,
                                                                        point_filter, ica_a),
                      list(range(ica_a.get_shape()[0].value)), initializer=term_acc_init)
    # term_1_acc, term_2_acc = loss_acc
    # return term_1 + term_1_acc, term_2 + term_2_acc
    print('called main sm fun')
    return loss_acc + tf.concat((term_1, term_2), axis=0)


def logistic_separable_nested_sm_map_fun(term_acc, index, x_tsr, a_k, w_k, depth_filter, point_filter, ica_a):
    w_l = get_single_separable_filter(depth_filter, point_filter, index)
    a_l = tf.slice(ica_a, [index], [1])
    # xw_l = tf.matmul(x_tsr, w_l)  # needs reviewing
    xw_l = tf.reduce_sum(x_tsr * w_l, axis=(1, 2))
    gp_l = -4.0 / tf.square(tf.exp(xw_l) + tf.exp(-xw_l))
    xw_k = tf.reduce_sum(x_tsr * w_k, axis=(1, 2))
    # xw_k = tf.matmul(x_tsr, w_k)  # needs reviewing
    gp_k = -4.0 / tf.square(tf.exp(xw_k) + tf.exp(-xw_k))

    ww_prod = tf.reduce_sum(w_k * w_l)
    gg_prod = tf.reduce_sum(gp_k * gp_l)
    term = a_k * a_l * ww_prod * ww_prod * gg_prod
    print('called nested sm fun')
    return term_acc + term


def logistic_ensemble_mrf_loss(xw, ica_a_flat):
    ica_a_pos = tf.nn.softplus(ica_a_flat)
    xw_abs = tf.abs(xw)
    log_sum_exp = tf.log(1 + tf.exp(-2 * xw_abs)) + xw_abs
    neg_g_wx = (tf.log(0.5) + log_sum_exp) * ica_a_pos
    neg_log_p_patches = tf.reduce_sum(neg_g_wx, axis=2)
    return tf.reduce_mean(neg_log_p_patches, axis=1, name='loss')


def student_ensemble_mrf_loss(xw, ica_a_flat):
    ica_a_pos = tf.nn.softplus(ica_a_flat)
    neg_g_wx = tf.log(1.0 + 0.5 * tf.square(xw)) * ica_a_pos
    neg_log_p_patches = tf.reduce_sum(neg_g_wx, axis=2)
    return tf.reduce_mean(neg_log_p_patches, axis=1, name='loss')


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
    # w_norm = tf.diag_part(ww_mat, name='w_norm')

    term_1 = tf.reduce_sum(ica_a_pos * gp_vec, name='t1')
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

    term_1 = tf.reduce_sum(tf.squeeze(ica_a_pos) * gp_vec, name='t1')
    term_2 = 0.5 * tf.reduce_sum(aa_mat * ww_mat * gg_mat, name='t2')
    return term_1 + term_2, term_1, term_2


def get_single_separable_filter(depth_filter, point_filter, index):
    # sum over dm(pf [DM, C, 1] x df [DM, 1, F]) = jk [C, F]

    filter_k = tf.matmul(tf.squeeze(tf.slice(point_filter, [index, 0, 0, 0], [1, -1, -1, -1]), axis=0), depth_filter)
    filter_k = tf.reduce_sum(filter_k, axis=0)
    norm_k = tf.norm(filter_k, ord=2)
    return filter_k / norm_k
