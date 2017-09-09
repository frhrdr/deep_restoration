import tensorflow as tf
import numpy as np
import time
import os
from modules.foe_full_prior import FoEFullPrior
from utils.temp_utils import patch_batch_gen, plot_img_mats, get_optimizer
from utils.preprocessing import make_data_dir
from utils.patch_prior_losses import logistic_channelwise_mrf_loss, logistic_channelwise_score_matching_loss, \
    student_channelwise_mrf_loss, student_channelwise_score_matching_loss
#  _mean_lc_sdev_none


class FoEChannelwisePrior(FoEFullPrior):
    """
    ICA prior which views each channel as independent
    """

    def __init__(self, tensor_names, weighting, classifier, filter_dims, input_scaling, n_components, n_channels,
                 n_features_white, dist='logistic', mean_mode='gc', sdev_mode='gc',
                 trainable=False, name=None, load_name=None, dir_name=None, load_tensor_names=None):

        super().__init__(tensor_names, weighting, classifier, filter_dims, input_scaling, n_components, n_channels,
                         n_features_white, dist=dist, mean_mode=mean_mode, sdev_mode=sdev_mode, trainable=trainable,
                         name=name, load_name=load_name, dir_name=dir_name, load_tensor_names=load_tensor_names)


    @staticmethod
    def assign_names(dist, name, load_name, dir_name, load_tensor_names, tensor_names):
        dist_options = ('student', 'logistic')
        assert dist in dist_options

        student_names = ('FoEStudentChannelwisePrior', 'FoEStudentChannelwisePrior', 'student_channelwise_prior')
        logistic_names = ('FoELogisticChannelwisePrior', 'FoELogisticChannelwisePrior', 'logistic_channelwise_prior')
        dist_names = student_names if dist == 'student' else logistic_names

        name = name if name is not None else dist_names[0]
        load_name = load_name if load_name is not None else dist_names[1]
        dir_name = dir_name if dir_name is not None else dist_names[2]
        load_tensor_names = load_tensor_names if load_tensor_names is not None else tensor_names

        return name, load_name, dir_name, load_tensor_names

    @staticmethod
    def get_load_path(dir_name, classifier, tensor_name, filter_dims, n_components,
                      n_features_white, mean_mode, sdev_mode):
        path = FoEFullPrior.get_load_path(dir_name, classifier, tensor_name, filter_dims, n_components,
                                          n_features_white, mean_mode, sdev_mode)
        return path.rstrip('/') + '_channelwise/'

    def mrf_loss(self, xw, ica_a_flat):
        if self.dist == 'logistic':
            return logistic_channelwise_mrf_loss(xw, ica_a_flat)
        elif self.dist == 'student':
            return student_channelwise_mrf_loss(xw, ica_a_flat)
        else:
            raise NotImplementedError

    def make_normed_filters(self, trainable, squeeze_alpha, add_to_var_list=True):
        ica_a = tf.get_variable('ica_a', shape=[self.n_channels, self.n_components, 1],
                                trainable=trainable, dtype=tf.float32)
        ica_w = tf.get_variable('ica_w', shape=[self.n_channels, self.n_features_white, self.n_components],
                                trainable=trainable, dtype=tf.float32)
        if add_to_var_list:
            self.var_list.extend([ica_a, ica_w])
        ica_w = ica_w / tf.stack([tf.norm(ica_w, ord=2, axis=1)] * self.n_features_white, axis=1)
        if squeeze_alpha:
            ica_a = tf.squeeze(ica_a)

        return ica_a, ica_w

    def build(self, scope_suffix=''):
        with tf.variable_scope(self.name):

            n_feats_per_channel = self.ph * self.pw
            whitening_tensor = tf.get_variable('whiten_mat',
                                               shape=[self.n_channels, self.n_features_white, n_feats_per_channel],
                                               dtype=tf.float32, trainable=False)
            ica_a = tf.get_variable('ica_a', shape=[self.n_channels, self.n_components, 1],
                                    trainable=self.trainable, dtype=tf.float32)
            ica_w = tf.get_variable('ica_w', shape=[self.n_channels, self.n_features_white, self.n_components],
                                    trainable=self.trainable, dtype=tf.float32)
            ica_a_squeezed = tf.squeeze(ica_a)

            whitened_mixing = tf.matmul(whitening_tensor, ica_w, transpose_a=True)

            if False and self.mean_mode in ('gc', 'gf') and self.sdev_mode in ('gc', 'gf'):
                normed_featmap = self.norm_feat_map_directly()  # shape [1, h, w, n_channels]
                print(whitened_mixing.get_shape())  # shape [n_channels, n_features_raw, n_components]
                whitened_mixing = tf.transpose(whitened_mixing, perm=[1, 0, 2])
                whitened_mixing = tf.reshape(whitened_mixing, [self.ph, self.pw,
                                                               self.n_channels, self.n_components])

                xw_stacked = tf.nn.depthwise_conv2d(normed_featmap, whitened_mixing,
                                                    strides=[1, 1, 1, 1], padding='VALID')
                # out shape [1, out_height, out_width, in_channels * channel_multiplier]
                bs, h, w, n_stacked_patches = [k.value for k in xw_stacked.get_shape()]
                assert bs == 1
                xw = tf.reshape(xw_stacked, shape=[h * w, self.n_channels, self.n_components])
                xw = tf.transpose(xw, perm=[1, 0, 2])

            else:
                normed_patches = self.shape_and_norm_tensor() # shape [n_patches, n_channels, n_feats_per_channel]
                xw = tf.matmul(tf.transpose(normed_patches, perm=[1, 0, 2]), whitened_mixing)

            self.loss = self.mrf_loss(xw, ica_a_squeezed)
            self.var_list.extend([ica_a, ica_w, whitening_tensor])

    def score_matching_loss(self, x_mat, ica_w, ica_a):
        if self.dist == 'logistic':
            return logistic_channelwise_score_matching_loss(x_mat, ica_w, ica_a)
        elif self.dist == 'student':
            return student_channelwise_score_matching_loss(x_mat, ica_w, ica_a)
        else:
            raise NotImplementedError

    def train_prior(self, batch_size, n_iterations, lr=3.0e-6, lr_lower_points=(), grad_clip=100.0, n_vis=144,
                    whiten_mode='pca', n_data_samples=100000,
                    n_val_samples=0,
                    log_freq=5000, summary_freq=10, print_freq=100, test_freq=0, prev_ckpt=0,
                    optimizer_name='adam',
                    plot_filters=False, do_clip=True):

        log_path = self.load_path

        data_dir = make_data_dir(in_tensor_name=self.in_tensor_names, ph=self.ph, pw=self.pw,
                                 mean_mode=self.mean_mode, sdev_mode=self.sdev_mode,
                                 n_features_white=self.n_features_white,
                                 classifier=self.classifier).rstrip('/') + '_channelwise/'

        data_gen = patch_batch_gen(batch_size, whiten_mode=whiten_mode, data_dir=data_dir,
                                   data_shape=(n_data_samples, self.n_channels, self.n_features_white))

        with tf.Graph().as_default() as graph:
            with tf.variable_scope(self.name):
                self.add_preprocessing_to_graph(data_dir, whiten_mode)

                x_pl = tf.placeholder(dtype=tf.float32, shape=[batch_size, self.n_channels, self.n_features_white],
                                      name='x_pl')
                x_tsr = tf.transpose(x_pl, perm=[1, 0, 2], name='x_mat')
                lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name='lr')

                ica_w = tf.get_variable(shape=[self.n_channels, self.n_features_white, self.n_components],
                                        dtype=tf.float32, name='ica_w',
                                        initializer=tf.random_normal_initializer(stddev=0.00001))
                ica_a = tf.get_variable(shape=[self.n_channels, self.n_components, 1], dtype=tf.float32,
                                        name='ica_a', initializer=tf.random_normal_initializer())

                loss, term_1, term_2 = self.score_matching_loss(x_mat=x_tsr, ica_w=ica_w, ica_a=ica_a)

                w_norm = tf.norm(ica_w, ord=2, axis=1)
                w_norm = tf.stack([tf.norm(ica_w, ord=2, axis=1)] * self.n_features_white, axis=1)
                clip_op = tf.assign(ica_w, ica_w / w_norm)
                opt = get_optimizer(name=optimizer_name, lr_pl=lr_pl)

                tvars = tf.trainable_variables()
                grads = tf.gradients(loss, tvars)
                tg_pairs = [k for k in zip(grads, tvars) if k[0] is not None]
                tg_clipped = [(tf.clip_by_value(k[0], -grad_clip, grad_clip), k[1])
                              for k in tg_pairs]
                opt_op = opt.apply_gradients(tg_clipped)

                if self.load_name != self.name and prev_ckpt:
                    to_load = self.tensor_load_dict_by_name(tf.global_variables())
                    saver = tf.train.Saver(var_list=to_load)
                else:
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
                    for count in range(prev_ckpt + 1, prev_ckpt + n_iterations + 1):
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
                            w_res, alp, t1, t2 = sess.run([ica_w, ica_a, term_1, term_2], feed_dict={x_pl: data})
                            print('it: ', count, ' / ', n_iterations + prev_ckpt)
                            print('mean a: ', np.mean(alp), ' max a: ', np.max(alp), ' min a: ', np.min(alp))
                            print('mean w: ', np.mean(w_res), ' max w: ', np.max(w_res), ' min w: ', np.min(w_res))
                            print('term_1: ', t1, ' term_2: ', t2)

                            train_ratio = 100.0 * train_time / (time.time() - start_time)
                            print('{0:2.1f}% of the time spent in run calls'.format(train_ratio))

                        if count % log_freq == 0:
                            saver.save(sess, checkpoint_file, write_meta_graph=False, global_step=count)

                    saver.save(sess, checkpoint_file, write_meta_graph=False, global_step=n_iterations)

                    if plot_filters:
                        unwhiten_mat = np.load(data_dir + 'unwhiten_' + whiten_mode + '.npy').astype(np.float32)
                        w_res, alp = sess.run([ica_w, ica_a])
                        print(alp)
                        comps = np.dot(w_res[0, :, :].T, unwhiten_mat[0, :, :])
                        # comps = np.concatenate((w_res[0, :, :], np.zeros(shape=[1, w_res.shape[2]])), axis=0).T
                        print(comps.shape)
                        comps -= np.min(comps)
                        comps /= np.max(comps)
                        co = np.reshape(comps[:n_vis, :], [-1, self.ph, self.pw])
                        plot_img_mats(co, color=False)

    def load_filters_for_plotting(self):
        with tf.Graph().as_default():
            with tf.variable_scope(self.name):
                n_features_raw = self.ph * self.pw * self.n_channels
                unwhitening_tensor = tf.get_variable('unwhiten_mat',
                                                     shape=[self.n_channels, self.n_features_white, n_features_raw],
                                                     dtype=tf.float32, trainable=False)
                self.var_list = [unwhitening_tensor]
                ica_a, ica_w = self.make_normed_filters(trainable=False, squeeze_alpha=True)

            with tf.Session() as sess:
                self.load_weights(sess)
                unwhitening_mat, a_mat, w_mat = sess.run([unwhitening_tensor, ica_a, ica_w])

        return unwhitening_mat, a_mat, w_mat

    def plot_filters_all_channels(self, channel_ids, save_path, save_as_mat=False, save_as_plot=True):
        """
        visualizes the patch for each channel of a trained filter and saves this as one plot.
        does so for the filter of each given index

        :param channel_ids: collection of channel indices
        :param save_path: location to save plots
        :param save_as_mat: if true, saves each filter as channel x height x width matrix
        :param save_as_plot:  if true, saves each filter as image
        :return None
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        unwhitening_mat, a_mat, w_mat = self.load_filters_for_plotting()
        w_mat_select = w_mat[channel_ids, :, :].transpose((0, 2, 1))
        rotated_w_mat = w_mat_select @ unwhitening_mat[channel_ids, :, :]

        for idx, channel_id in enumerate(channel_ids):
            chan_filters = rotated_w_mat[idx, :, :]
            plottable_filters = np.reshape(chan_filters, [self.n_components, self.ph, self.pw])

            if save_as_mat:
                file_name = 'channel_{}_filters.npy'.format(channel_id)
                np.save(save_path + file_name, plottable_filters)
            if save_as_plot:
                file_name = 'channel_{}_filters.png'.format(channel_id)
                plottable_filters -= np.min(plottable_filters)
                plottable_filters /= np.max(plottable_filters)
                plot_img_mats(plottable_filters, rescale=False, show=False, save_path=save_path + file_name)
                print('channel {} done'.format(channel_id))
