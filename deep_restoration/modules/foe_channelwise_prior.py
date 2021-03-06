import tensorflow as tf
import numpy as np
import os
from modules.foe_full_prior import FoEFullPrior
from utils.temp_utils import plot_img_mats
from utils.patch_prior_losses import logistic_channelwise_mrf_loss, logistic_channelwise_score_matching_loss, \
    student_channelwise_mrf_loss, student_channelwise_score_matching_loss
#  _mean_lc_sdev_none


class FoEChannelwisePrior(FoEFullPrior):
    """
    FoE prior which views each channel as independent
    """

    def __init__(self, tensor_names, weighting, classifier, filter_dims, input_scaling, n_components, n_channels,
                 n_features_per_channel_white, dist='logistic', mean_mode='gc', sdev_mode='gc', whiten_mode='pca',
                 name=None, load_name=None, dir_name=None, load_tensor_names=None):

        n_features_white = n_features_per_channel_white * n_channels
        super().__init__(tensor_names, weighting, classifier, filter_dims, input_scaling, n_components, n_channels,
                         n_features_per_channel_white, dist=dist,
                         mean_mode=mean_mode, sdev_mode=sdev_mode, whiten_mode=whiten_mode,
                         name=name, load_name=load_name, dir_name=dir_name, load_tensor_names=load_tensor_names)

        self.n_features_per_channel_white = n_features_per_channel_white
        self.n_features_white = n_features_white

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
                      n_features_per_channel_white, mean_mode, sdev_mode, whiten_mode):
        path = FoEFullPrior.get_load_path(dir_name, classifier, tensor_name, filter_dims, n_components,
                                          n_features_per_channel_white, mean_mode, sdev_mode, whiten_mode)
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
                                trainable=trainable, dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=1.0))
        ica_w = tf.get_variable('ica_w', shape=[self.n_channels, self.n_features_per_channel_white, self.n_components],
                                trainable=trainable, dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=0.001))
        if add_to_var_list:
            self.var_list.extend([ica_a, ica_w])

        w_norm = tf.stack([tf.norm(ica_w, ord=2, axis=1)] * self.n_features_per_channel_white, axis=1)
        ica_w = ica_w / w_norm

        if squeeze_alpha:
            ica_a = tf.squeeze(ica_a)

        extra_op = None
        extra_vals = None
        return ica_a, ica_w, extra_op, extra_vals

    def build(self, scope_suffix='', featmap_tensor=None):
        with tf.variable_scope(self.name):

            whitening_tensor = tf.get_variable('whiten_mat',
                                               shape=[self.n_channels, self.n_features_per_channel_white,
                                                      self.ph * self.pw],
                                               dtype=tf.float32, trainable=False)

            ica_a, ica_w, _, _ = self.make_normed_filters(trainable=False, squeeze_alpha=True)

            whitened_mixing = tf.matmul(whitening_tensor, ica_w, transpose_a=True)
            featmap = self.get_tensors() if featmap_tensor is None else featmap_tensor

            if False and self.mean_mode in ('gc', 'gf') and self.sdev_mode in ('gc', 'gf'):
                normed_featmap = self.norm_feat_map_directly(featmap)  # shape [1, h, w, n_channels]
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
                normed_patches = self.shape_and_norm_featmap(featmap)  # shape [n_patches, n_channels, n_feats_per_channel]
                xw = tf.matmul(tf.transpose(normed_patches, perm=[1, 0, 2]), whitened_mixing)

            self.loss = self.mrf_loss(xw, ica_a)
            self.var_list.append(whitening_tensor)

    def score_matching_loss(self, x_mat, ica_w, ica_a):
        x_mat = tf.transpose(x_mat, perm=[1, 0, 2], name='x_mat')
        if self.dist == 'logistic':
            return logistic_channelwise_score_matching_loss(x_mat, ica_w, ica_a)
        elif self.dist == 'student':
            return student_channelwise_score_matching_loss(x_mat, ica_w, ica_a)
        else:
            raise NotImplementedError

    def make_data_dir(self):
        n_features_white = self.n_features_white
        self.n_features_white = self.n_features_per_channel_white
        data_dir = super().make_data_dir()
        self.n_features_white = n_features_white
        return data_dir.rstrip('/') + '_channelwise/'

    def get_x_placeholder(self, batch_size):
        shape = [batch_size, self.n_channels, self.n_features_per_channel_white]
        return tf.placeholder(dtype=tf.float32, shape=shape, name='x_pl')

    def plot_filters_after_training(self, w_res, unwhiten_mat, n_vis):
        comps = np.dot(w_res[0, :, :].T, unwhiten_mat[0, :, :])
        print(comps.shape)
        comps -= np.min(comps)
        comps /= np.max(comps)
        co = np.reshape(comps[:n_vis, :], [-1, self.ph, self.pw])
        plot_img_mats(co, color=False, rescale=True)

    def load_filters_for_plotting(self):
        with tf.Graph().as_default():
            with tf.variable_scope(self.name):
                n_features_raw = self.ph * self.pw
                unwhitening_tensor = tf.get_variable('unwhiten_mat',
                                                     shape=[self.n_channels, self.n_features_per_channel_white,
                                                            n_features_raw],
                                                     dtype=tf.float32, trainable=False)
                self.var_list = [unwhitening_tensor]
                ica_a, ica_w, _, _ = self.make_normed_filters(trainable=False, squeeze_alpha=True)

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

    def patch_batch_gen(self, batch_size, data_dir, n_samples, data_mode='train'):
        assert data_mode in ('train', 'validate')

        if data_mode == 'train':
            data_mat = np.memmap(data_dir + 'data_mat_' + self.whiten_mode + '_whitened_channelwise.npy',
                                 dtype=np.float32, mode='r', shape=(n_samples, self.n_channels,
                                                                    self.n_features_per_channel_white))

            idx = 0
            while True:
                if idx + batch_size < n_samples:
                    batch = data_mat[idx:(idx + batch_size), :, :]
                    idx += batch_size
                else:
                    last_bit = data_mat[idx:, :, :]
                    idx = (idx + batch_size) % n_samples
                    first_bit = data_mat[:idx, :, :]
                    batch = np.concatenate((last_bit, first_bit), axis=0)
                yield batch
        else:
            data_mat = np.load(data_dir + 'val_mat.npy')
            assert data_mat.shape[0] == n_samples, 'expected {}, found {}'.format(n_samples, data_mat.shape[0])
            assert n_samples % batch_size == 0

            idx = 0
            while True:
                batch = data_mat[idx:(idx + batch_size), :, :]
                idx += batch_size
                idx = idx % n_samples
                yield batch

    # def train_prior(self, batch_size, n_iterations, lr=3.0e-6, lr_lower_points=(), grad_clip=100.0, n_vis=144,
    #                 n_data_samples=100000, n_val_samples=0,
    #                 log_freq=5000, summary_freq=10, print_freq=100, test_freq=0, prev_ckpt=0,
    #                 optimizer_name='adam',
    #                 plot_filters=False, do_clip=True):
    #
    #     if not os.path.exists(self.load_path):
    #         os.makedirs(self.load_path)
    #
    #     data_dir = self.make_data_dir()
    #     data_gen = self.patch_batch_gen(batch_size, data_dir=data_dir, n_samples=n_data_samples, data_mode='train')
    #     val_gen = self.patch_batch_gen(batch_size, data_dir=data_dir, n_samples=n_val_samples, data_mode='validate')
    #
    #     with tf.Graph().as_default() as graph:
    #         with tf.variable_scope(self.name):
    #             self.add_preprocessing_to_graph(data_dir, self.whiten_mode)
    #
    #             ica_a, ica_w, extra_op, _ = self.make_normed_filters(trainable=True, squeeze_alpha=False)
    #
    #             x_pl = self.get_x_placeholder(batch_size)
    #             loss, term_1, term_2 = self.score_matching_loss(x_mat=x_pl, ica_w=ica_w, ica_a=ica_a)
    #
    #             lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name='lr')
    #             opt = get_optimizer(name=optimizer_name, lr_pl=lr_pl)
    #             tvars = tf.trainable_variables()
    #             grads = tf.gradients(loss, tvars)
    #             tg_pairs = [k for k in zip(grads, tvars) if k[0] is not None]
    #             tg_clipped = [(tf.clip_by_value(k[0], -grad_clip, grad_clip), k[1])
    #                           for k in tg_pairs]
    #             opt_op = opt.apply_gradients(tg_clipped)
    #
    #             if self.load_name != self.name and prev_ckpt:
    #                 to_load = self.tensor_load_dict_by_name(tf.global_variables())
    #                 saver = tf.train.Saver(var_list=to_load)
    #             else:
    #                 saver = tf.train.Saver()
    #
    #             checkpoint_file = os.path.join(self.load_path, 'ckpt')
    #
    #             if not os.path.exists(self.load_path):
    #                 os.makedirs(self.load_path)
    #
    #             tf.summary.scalar('total_loss', loss)
    #             tf.summary.scalar('term_1', term_1)
    #             tf.summary.scalar('term_2', term_2)
    #             train_summary_op = tf.summary.merge_all()
    #             summary_writer = tf.summary.FileWriter(self.load_path + '/summaries')
    #
    #             val_loss = tf.placeholder(dtype=tf.float32, shape=[], name='val_loss')
    #             val_summary_op = tf.summary.scalar('validation_loss', val_loss)
    #
    #             with tf.Session() as sess:
    #                 sess.run(tf.global_variables_initializer())
    #
    #                 if prev_ckpt:
    #                     self.load_weights(sess, prev_ckpt)
    #
    #                 start_time = time.time()
    #                 train_time = 0
    #                 for count in range(prev_ckpt + 1, prev_ckpt + n_iterations + 1):
    #                     data = next(data_gen)
    #
    #                     if lr_lower_points and lr_lower_points[0][0] <= count:
    #                         lr = lr_lower_points[0][1]
    #                         print('new learning rate: ', lr)
    #                         lr_lower_points = lr_lower_points[1:]
    #
    #                     batch_start = time.time()
    #                     batch_loss, _, summary_string = sess.run(fetches=[loss, opt_op, train_summary_op],
    #                                                              feed_dict={x_pl: data, lr_pl: lr})
    #
    #                     train_time += time.time() - batch_start
    #
    #                     if count % summary_freq == 0:
    #                         summary_writer.add_summary(summary_string, count)
    #
    #                     if count % print_freq == 0:
    #                         print(batch_loss)
    #
    #                     if count % (print_freq * 10) == 0:
    #                         term_1 = graph.get_tensor_by_name(self.name + '/t1:0')
    #                         term_2 = graph.get_tensor_by_name(self.name + '/t2:0')
    #                         w_res, alp, t1, t2 = sess.run([ica_w, ica_a, term_1, term_2], feed_dict={x_pl: data})
    #                         print('it: ', count, ' / ', n_iterations + prev_ckpt)
    #                         print('mean a: ', np.mean(alp), ' max a: ', np.max(alp), ' min a: ', np.min(alp))
    #                         print('mean w: ', np.mean(w_res), ' max w: ', np.max(w_res), ' min w: ', np.min(w_res))
    #                         print('term_1: ', t1, ' term_2: ', t2)
    #
    #                         train_ratio = 100.0 * train_time / (time.time() - start_time)
    #                         print('{0:2.1f}% of the time spent in run calls'.format(train_ratio))
    #
    #                     if test_freq > 0 and count % test_freq == 0:
    #                         val_loss_acc = 0.0
    #                         num_runs = n_val_samples // batch_size
    #                         for val_count in range(num_runs):
    #                             val_feed_dict = {x_pl: next(val_gen)}
    #                             val_batch_loss = sess.run(loss, feed_dict=val_feed_dict)
    #                             val_loss_acc += val_batch_loss
    #                         val_loss_acc /= num_runs
    #                         val_summary_string = sess.run(val_summary_op, feed_dict={val_loss: val_loss_acc})
    #                         summary_writer.add_summary(val_summary_string, count)
    #                         print(('Iteration: {0:6d} Validation Error: {1:9.2f} ' +
    #                                'Time: {2:5.1f} min').format(count, val_loss_acc,
    #                                                             (time.time() - start_time) / 60))
    #
    #                     if count % log_freq == 0:
    #                         if extra_op is not None:
    #                             sess.run(extra_op)
    #                         saver.save(sess, checkpoint_file, write_meta_graph=False, global_step=count)
    #
    #                 saver.save(sess, checkpoint_file, write_meta_graph=False, global_step=n_iterations + prev_ckpt)
    #
    #                 if plot_filters:
    #                     unwhiten_mat = np.load(data_dir + 'unwhiten_' + self.whiten_mode + '.npy').astype(np.float32)
    #                     w_res = sess.run(ica_w)
    #
    #                     self.plot_filters_after_training(w_res, unwhiten_mat, n_vis)
