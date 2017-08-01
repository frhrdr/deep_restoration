import tensorflow as tf
import numpy as np
import time
import os
import matplotlib
matplotlib.use('tkagg', force=True)
from utils.temp_utils import plot_img_mats, patch_batch_gen


def score_matching_loss(x_mat, w_mat, alpha):
    const_T = x_mat.get_shape()[0].value
    xw_mat = tf.matmul(x_mat, w_mat)
    g_mat = -tf.tanh(xw_mat)
    gp_mat = -4.0 / tf.square(tf.exp(xw_mat) + tf.exp(-xw_mat))  # d/dx tanh(x) = 4 / (exp(x) + exp(-x))^2
    gp_vec = tf.reduce_sum(gp_mat, axis=0) / const_T
    gg_mat = tf.matmul(g_mat, g_mat, transpose_a=True) / const_T
    aa_mat = tf.matmul(alpha, alpha, transpose_b=True)
    ww_mat = tf.matmul(w_mat, w_mat, transpose_a=True)
    w_norm = tf.diag_part(ww_mat, name='w_norm')

    term_1 = tf.reduce_sum(alpha * w_norm * gp_vec, name='t1')
    term_2 = 0.5 * tf.reduce_sum(aa_mat * ww_mat * gg_mat, name='t2')
    return term_1 + term_2


def train_test():
    ph = 8
    pw = 8
    color = True
    n_features = ph * pw
    if color:
        n_features *= 3
    n_features -= 1
    n_components = 512
    batch_size = 1000
    num_iterations = 20000
    lr = 3.0e-6
    lr_lower_points = [(0, 1.0e-3), (2000, 1.0e-4), (4000, 3.0e-5), (10000, 1.0e-5)]
    grad_clip = 100.0
    n_vis = 144
    whiten_mode = 'pca'
    data_dir = './data/patches_color/8x8/'
    log_path = './logs/priors/ica_prior/color_8x8_1024comp/'
    data_gen = patch_batch_gen(1000, whiten_mode=whiten_mode, data_dir=data_dir, data_shape=(100000, n_features))
    unwhiten_mat = np.load(data_dir + 'unwhiten_' + whiten_mode + '.npy').astype(np.float32)
    whiten_mat = np.load(data_dir + 'whiten_' + whiten_mode + '.npy').astype(np.float32)
    with tf.Graph().as_default() as graph:
        with tf.variable_scope('ICAPrior'):
            # add whitening mats to the save-files for later retrieval
            tf.get_variable(name='whiten_mat', initializer=whiten_mat, trainable=False, dtype=tf.float32)
            tf.get_variable(name='unwhiten_mat', initializer=unwhiten_mat, trainable=False, dtype=tf.float32)

            x_pl = tf.placeholder(dtype=tf.float32, shape=[batch_size, n_features], name='x_pl')
            lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name='lr')

            w_mat = tf.get_variable(shape=[n_features, n_components], dtype=tf.float32, name='ica_w',
                                    initializer=tf.random_normal_initializer(stddev=0.001))
            alpha = tf.get_variable(shape=[n_components, 1], dtype=tf.float32, name='ica_a',
                                    initializer=tf.random_normal_initializer())

            loss = score_matching_loss(x_mat=x_pl, w_mat=w_mat, alpha=alpha)
            clip_op = tf.assign(w_mat, w_mat / tf.norm(w_mat, ord=2, axis=0))

            opt = tf.train.MomentumOptimizer(learning_rate=lr_pl, momentum=0.9)
            tvars = tf.trainable_variables()
            grads = tf.gradients(loss, tvars)
            tg_pairs = [k for k in zip(grads, tvars) if k[0] is not None]
            tg_clipped = [(tf.clip_by_value(k[0], -grad_clip, grad_clip), k[1])
                          for k in tg_pairs]
            opt_op = opt.apply_gradients(tg_clipped)

            saver = tf.train.Saver()

            with tf.Session() as sess:

                sess.run(tf.global_variables_initializer())
                sess.run(clip_op)

                start_time = time.time()
                train_time = 0
                for idx in range(num_iterations):
                    data = next(data_gen)

                    if lr_lower_points and lr_lower_points[0][0] == idx:
                        lr = lr_lower_points[0][1]
                        print('new learning rate: ', lr)
                        lr_lower_points = lr_lower_points[1:]
                    batch_start = time.time()
                    batch_loss, _ = sess.run(fetches=[loss, opt_op], feed_dict={x_pl: data, lr_pl: lr})
                    sess.run(clip_op)
                    train_time += time.time() - batch_start
                    if idx % 100 == 0:
                        print(batch_loss)

                    if idx % 1000 == 0:
                        term1 = graph.get_tensor_by_name('ICAPrior/t1:0')
                        term2 = graph.get_tensor_by_name('ICAPrior/t2:0')
                        w_res, alp, t1, t2 = sess.run([w_mat, alpha, term1, term2], feed_dict={x_pl: data})
                        print('it: ', idx, ' / ', num_iterations)
                        print('mean a: ', np.mean(alp), ' max a: ', np.max(alp), ' min a: ', np.min(alp))
                        print('mean w: ', np.mean(w_res), ' max w: ', np.max(w_res), ' min w: ', np.min(w_res))
                        print('term1: ', t1, ' term2: ', t2)

                        train_ratio = 100.0 * train_time / (time.time() - start_time)
                        print('{0:2.1f}% of the time spent in run calls'.format(train_ratio))

                checkpoint_file = os.path.join(log_path, 'ckpt')
                saver.save(sess, checkpoint_file, write_meta_graph=False, global_step=num_iterations)

                w_res, alp = sess.run([w_mat, alpha])
                print(alp)
                comps = np.dot(w_res.T, unwhiten_mat)
                print(comps.shape)
                comps -= np.min(comps)
                comps /= np.max(comps)
                if color:
                    co = np.reshape(comps[:n_vis, :], [-1, ph, pw, 3])
                else:
                    co = np.reshape(comps[:n_vis, :], [-1, ph, pw])
                plot_img_mats(co, color=color)

