import tensorflow as tf
import numpy as np
from modules.ica_prior import ICAPrior
from utils.temp_utils import patch_batch_gen, plot_img_mats, get_optimizer
from utils.preprocessing import make_data_dir
import time
import os
#  _mean_lc_sdev_none


class ChannelICAPrior(ICAPrior):
    """
    ICA prior which views each channel as independent
    """

    def __init__(self, tensor_names, weighting, classifier, filter_dims, input_scaling, n_components, n_channels,
                 n_features_white,
                 trainable=False, name='ChannelICAPrior', load_name='ChannelICAPrior', dir_name='channel_ica_prior',
                 mean_mode='lc', sdev_mode='none', load_tensor_names=None):
        super().__init__(tensor_names, weighting, classifier, filter_dims, input_scaling, n_components, n_channels,
                         n_features_white, trainable=trainable, name=name, load_name=load_name, dir_name=dir_name,
                         mean_mode=mean_mode, sdev_mode=sdev_mode, load_tensor_names=load_tensor_names)
        self.n_features_total = n_features_white * n_channels
        self.load_path = self.load_path.rstrip('/') + '_channelwise/'

    @staticmethod
    def mrf_loss(xw, ica_a_squeezed):
        xw_abs = tf.abs(xw)
        log_sum_exp = tf.log(1 + tf.exp(-2 * xw_abs)) + xw_abs
        neg_g_wx = (tf.log(0.5) + log_sum_exp) * tf.stack([ica_a_squeezed] * xw.get_shape()[1].value, axis=1)
        neg_log_p_patches = tf.reduce_sum(neg_g_wx, axis=1)
        return tf.reduce_mean(neg_log_p_patches, name='loss')

    def build(self, scope_suffix=''):
        with tf.variable_scope(self.name):

            n_feats_per_channel = self.filter_dims[0] * self.filter_dims[1]
            whitening_tensor = tf.get_variable('whiten_mat',
                                               shape=[self.n_channels, self.n_features_white, n_feats_per_channel],
                                               dtype=tf.float32, trainable=False)
            ica_a = tf.get_variable('ica_a', shape=[self.n_channels, self.n_components, 1],
                                    trainable=self.trainable, dtype=tf.float32)
            ica_w = tf.get_variable('ica_w', shape=[self.n_channels, self.n_features_white, self.n_components],
                                    trainable=self.trainable, dtype=tf.float32)
            ica_a_squeezed = tf.squeeze(tf.nn.softplus(ica_a))

            whitened_mixing = tf.matmul(whitening_tensor, ica_w, transpose_a=True)

            if False and self.mean_mode in ('gc', 'gf') and self.sdev_mode in ('gc', 'gf'):
                normed_featmap = self.norm_feat_map_directly()  # shape [1, h, w, n_channels]
                print(whitened_mixing.get_shape())  # shape [n_channels, n_features_raw, n_components]
                whitened_mixing = tf.transpose(whitened_mixing, perm=[1, 0, 2])
                whitened_mixing = tf.reshape(whitened_mixing, [self.filter_dims[0], self.filter_dims[1],
                                                               self.n_channels, self.n_components])

                xw_stacked = tf.nn.depthwise_conv2d(normed_featmap, whitened_mixing,
                                                    strides=[1, 1, 1, 1], padding='VALID')
                # out shape [1, out_height, out_width, in_channels * channel_multiplier]
                bs, h, w, n_stacked_patches = [k.value for k in xw_stacked.get_shape()]
                assert bs == 1
                xw = tf.reshape(xw_stacked, shape=[h * w, self.n_channels, self.n_components])
                xw = tf.transpose(xw, perm=[1, 0, 2])

                # whitened_mixing = tf.reshape(whitened_mixing, shape=[self.n_channels, self.filter_dims[0],
                #                                                      self.filter_dims[1], self.n_components])
                # whitened_mixing = tf.transpose(whitened_mixing, perm=[1, 2, 0, 3])
                # print(normed_featmap.get_shape())
                # print(whitened_mixing.get_shape())
                # xw = tf.nn.conv2d(normed_featmap, whitened_mixing, strides=[1, 1, 1, 1], padding='VALID')
                # xw = tf.reshape(xw, shape=[-1, self.n_components])
            else:
                normed_patches = self.shape_and_norm_tensor() # shape [n_patches, n_channels, n_feats_per_channel]
                xw = tf.matmul(tf.transpose(normed_patches, perm=[1, 0, 2]), whitened_mixing)

            self.loss = self.mrf_loss(xw, ica_a_squeezed)
            self.var_list.extend([ica_a, ica_w, whitening_tensor])

    @staticmethod
    def score_matching_loss(x_mat, w_mat, alpha):
        """
        :param x_mat: data    -- n_channels x batch_size x n_features_per_channel_white
        :param w_mat: filters -- n_channels x n_features_per_channel_white x n_components_per_channel
        :param alpha: scaling -- n_channels x n_components_per_channel
        :return: full loss, and two partial loss terms
        """
        alpha_pos = tf.nn.softplus(alpha)
        const_t = x_mat.get_shape()[0].value
        xw_mat = tf.matmul(x_mat, w_mat)
        g_mat = -tf.tanh(xw_mat)
        gp_mat = -4.0 / tf.square(tf.exp(xw_mat) + tf.exp(-xw_mat))  # d/dx tanh(x) = 4 / (exp(x) + exp(-x))^2
        gp_vec = tf.reduce_sum(gp_mat, axis=1) / const_t
        gg_mat = tf.matmul(g_mat, g_mat, transpose_a=True) / const_t
        aa_mat = tf.matmul(alpha_pos, alpha_pos, transpose_b=True)
        ww_mat = tf.matmul(w_mat, w_mat, transpose_a=True)

        ww_list = tf.split(ww_mat, num_or_size_splits=ww_mat.get_shape()[0].value, axis=0)
        w_norm_list = [tf.diag_part(tf.squeeze(k)) for k in ww_list]
        w_norm = tf.stack(w_norm_list, axis=0, name='w_norm')

        term_1 = tf.reduce_sum(tf.squeeze(alpha_pos) * w_norm * gp_vec, name='t1')
        term_2 = 0.5 * tf.reduce_sum(aa_mat * ww_mat * gg_mat, name='t2')
        return term_1 + term_2, term_1, term_2

    def train_prior(self, batch_size, num_iterations, lr=3.0e-6, lr_lower_points=(), grad_clip=100.0, n_vis=144,
                    whiten_mode='pca', num_data_samples=100000,
                    n_val_samples=0,
                    log_freq=5000, summary_freq=10, print_freq=100, test_freq=0, prev_ckpt=0,
                    optimizer_name='adam',
                    plot_filters=False, do_clip=True):

        log_path = self.load_path
        ph, pw = self.filter_dims

        data_dir = make_data_dir(in_tensor_name=self.in_tensor_names, ph=ph, pw=pw,
                                 mean_mode=self.mean_mode, sdev_mode=self.sdev_mode,
                                 n_features_white=self.n_features_white,
                                 classifier=self.classifier).rstrip('/') + '_channelwise/'

        data_gen = patch_batch_gen(batch_size, whiten_mode=whiten_mode, data_dir=data_dir,
                                   data_shape=(num_data_samples, self.n_channels, self.n_features_white))

        with tf.Graph().as_default() as graph:
            with tf.variable_scope(self.name):
                self.add_preprocessing_to_graph(data_dir, whiten_mode)

                x_pl = tf.placeholder(dtype=tf.float32, shape=[batch_size, self.n_channels, self.n_features_white],
                                      name='x_pl')
                x_mat = tf.transpose(x_pl, perm=[1, 0, 2], name='x_mat')
                lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name='lr')

                w_mat = tf.get_variable(shape=[self.n_channels, self.n_features_white, self.n_components],
                                        dtype=tf.float32, name='ica_w',
                                        initializer=tf.random_normal_initializer(stddev=0.00001))
                alpha = tf.get_variable(shape=[self.n_channels, self.n_components, 1], dtype=tf.float32,
                                        name='ica_a', initializer=tf.random_normal_initializer())

                loss, term_1, term_2 = self.score_matching_loss(x_mat=x_mat, w_mat=w_mat, alpha=alpha)

                w_norm = tf.norm(w_mat, ord=2, axis=1)
                w_norm = tf.stack([w_norm] * self.n_features_white, axis=1)
                clip_op = tf.assign(w_mat, w_mat / w_norm)
                opt = get_optimizer(name=optimizer_name, lr_pl=lr_pl)

                tvars = tf.trainable_variables()
                grads = tf.gradients(loss, tvars)
                tg_pairs = [k for k in zip(grads, tvars) if k[0] is not None]
                tg_clipped = [(tf.clip_by_value(k[0], -grad_clip, grad_clip), k[1])
                              for k in tg_pairs]
                opt_op = opt.apply_gradients(tg_clipped)

                saver = tf.train.Saver()

                checkpoint_file = os.path.join(log_path, 'ckpt')

                if not os.path.exists(log_path):
                    os.makedirs(log_path)

                tf.summary.scalar('total_loss', loss)
                tf.summary.scalar('term_1', term_1)
                tf.summary.scalar('term_2', term_2)
                train_summary_op = tf.summary.merge_all()
                summary_writer = tf.summary.FileWriter(log_path + '/summaries')

                with tf.Session() as sess:

                    sess.run(tf.global_variables_initializer())

                    if prev_ckpt:
                        prev_path = self.load_path + 'ckpt-' + str(prev_ckpt)
                        saver.restore(sess, prev_path)

                    sess.run(clip_op)
                    # plot_img_mats(whiten_mat[0, :, :].reshape([-1, 8, 8]), color=False, rescale=True)
                    # print(whiten_mat[0, :, :] @ unwhiten_mat[0, :, :].T)
                    start_time = time.time()
                    train_time = 0
                    for count in range(prev_ckpt + 1, prev_ckpt + num_iterations + 1):
                        data = next(data_gen)

                        if lr_lower_points and lr_lower_points[0][0] <= count:
                            lr = lr_lower_points[0][1]
                            print('new learning rate: ', lr)
                            lr_lower_points = lr_lower_points[1:]

                        batch_start = time.time()
                        batch_loss, _, summary_string = sess.run(fetches=[loss, opt_op, train_summary_op],
                                                                 feed_dict={x_pl: data, lr_pl: lr})
                        sess.run(clip_op)
                        train_time += time.time() - batch_start

                        if count % summary_freq == 0:
                            summary_writer.add_summary(summary_string, count)

                        if count % print_freq == 0:
                            print(batch_loss)

                        if count % (print_freq * 10) == 0:
                            term_1 = graph.get_tensor_by_name(self.name + '/t1:0')
                            term_2 = graph.get_tensor_by_name(self.name + '/t2:0')
                            w_res, alp, t1, t2 = sess.run([w_mat, alpha, term_1, term_2], feed_dict={x_pl: data})
                            print('it: ', count, ' / ', num_iterations + prev_ckpt)
                            print('mean a: ', np.mean(alp), ' max a: ', np.max(alp), ' min a: ', np.min(alp))
                            print('mean w: ', np.mean(w_res), ' max w: ', np.max(w_res), ' min w: ', np.min(w_res))
                            print('term_1: ', t1, ' term_2: ', t2)

                            train_ratio = 100.0 * train_time / (time.time() - start_time)
                            print('{0:2.1f}% of the time spent in run calls'.format(train_ratio))

                        if count % log_freq == 0:
                            saver.save(sess, checkpoint_file, write_meta_graph=False, global_step=count)

                    saver.save(sess, checkpoint_file, write_meta_graph=False, global_step=num_iterations)

                    if plot_filters:
                        unwhiten_mat = np.load(data_dir + 'unwhiten_' + whiten_mode + '.npy').astype(np.float32)
                        w_res, alp = sess.run([w_mat, alpha])
                        print(alp)
                        comps = np.dot(w_res[0, :, :].T, unwhiten_mat[0, :, :])
                        # comps = np.concatenate((w_res[0, :, :], np.zeros(shape=[1, w_res.shape[2]])), axis=0).T
                        print(comps.shape)
                        comps -= np.min(comps)
                        comps /= np.max(comps)
                        co = np.reshape(comps[:n_vis, :], [-1, ph, pw])
                        plot_img_mats(co, color=False)

    def plot_filters_all_channels(self, channel_ids, save_path, save_as_mat=False, save_as_plot=True):
        """
        visualizes the patch for each channel of a trained filter and saves this as one plot.
        does so for the filter of each given index

        :param filter_ids: collection of filter indices
        :param save_path: location to save plots
        :param save_as_mat: if true, saves each filter as channel x height x width matrix
        :param save_as_plot:  if true, saves each filter as image
        :return None
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with tf.Graph().as_default():

            with tf.variable_scope(self.name):
                n_features_raw = self.filter_dims[0] * self.filter_dims[1]
                unwhitening_tensor = tf.get_variable('unwhiten_mat', shape=[self.n_channels, self.n_features_white, n_features_raw],
                                                     dtype=tf.float32, trainable=False)
                ica_a = tf.get_variable('ica_a', shape=[self.n_channels, self.n_components, 1], trainable=self.trainable,
                                        dtype=tf.float32)
                ica_w = tf.get_variable('ica_w', shape=[self.n_channels, self.n_features_white, self.n_components],
                                        trainable=self.trainable, dtype=tf.float32)
            self.var_list = [unwhitening_tensor, ica_a, ica_w]

            with tf.Session() as sess:
                self.load_weights(sess)
                unwhitening_mat, a_mat, w_mat = sess.run(self.var_list)

            print('matrices loaded')
            # w_mat [n_chan, n_fw, n_comp], unwhiten_mat [n_chan, n_fw, n_f] -> [n_chan_select, n_comps, n_f]
            w_mat_select = w_mat[channel_ids, :, :].transpose((0, 2, 1))
            rotated_w_mat = w_mat_select @ unwhitening_mat[channel_ids, :, :]

            print('whitening reversed')

            for idx, channel_id in enumerate(channel_ids):
                chan_filters = rotated_w_mat[idx, :, :]
                plottable_filters = np.reshape(chan_filters, [self.n_components, self.filter_dims[0], self.filter_dims[1]])

                if save_as_mat:
                    file_name = 'channel_{}_filters.npy'.format(channel_id)
                    np.save(save_path + file_name, plottable_filters)
                if save_as_plot:
                    file_name = 'channel_{}_filters.png'.format(channel_id)
                    plottable_filters -= np.min(plottable_filters)
                    plottable_filters /= np.max(plottable_filters)
                    plot_img_mats(plottable_filters, rescale=False, show=False, save_path=save_path + file_name)
                    print('channel {} done'.format(channel_id))
