import tensorflow as tf
import numpy as np
import time
import os
from modules.core_modules import LearnedPriorLoss
from utils.temp_utils import flattening_filter, plot_img_mats, get_optimizer
from utils.preprocessing import preprocess_patch_tensor, preprocess_featmap_tensor
from utils.patch_prior_losses import logistic_full_mrf_loss, logistic_full_score_matching_loss, \
    student_full_mrf_loss, student_full_score_matching_loss


class FoEFullPrior(LearnedPriorLoss):
    def __init__(self, tensor_names, weighting, classifier, filter_dims, input_scaling,
                 n_components, n_channels, n_features_white,
                 dist='student', mean_mode='gc', sdev_mode='gc', whiten_mode='pca',
                 name=None, load_name=None, dir_name=None, load_tensor_names=None):

        name, load_name, dir_name, load_tensor_names = self.assign_names(dist, name, load_name, dir_name,
                                                                         load_tensor_names, tensor_names)

        mode_abbreviatons = {'global_channel': 'gc', 'global_feature': 'gf', 'local_channel': 'lc', 'local_full': 'lf'}
        if mean_mode in mode_abbreviatons:
            mean_mode = mode_abbreviatons[mean_mode]
        if sdev_mode in mode_abbreviatons:
            sdev_mode = mode_abbreviatons[sdev_mode]

        load_path = self.get_load_path(dir_name, classifier, load_tensor_names, filter_dims,
                                       n_components, n_features_white, mean_mode, sdev_mode, whiten_mode)

        super().__init__(tensor_names, weighting, name, load_path, load_name)
        # self.filter_dims = filter_dims  # tuple (height, width)
        self.ph = filter_dims[0]
        self.pw = filter_dims[1]
        self.input_scaling = input_scaling  # most likely 1, 255 or 1/255
        self.n_components = n_components  # number of components to be produced
        self.n_channels = n_channels
        self.n_features_white = n_features_white
        self.dir_name = dir_name
        self.classifier = classifier
        self.mean_mode = mean_mode
        self.sdev_mode = sdev_mode
        self.whiten_mode = whiten_mode
        self.dist = dist

        self.min_val_loss = float('inf')
        self.increasing_val_flag = 0

    @staticmethod
    def assign_names(dist, name, load_name, dir_name, load_tensor_names, tensor_names):
        dist_options = ('student', 'logistic')
        assert dist in dist_options

        student_names = ('FoEStudentFullPrior', 'FoEStudentFullPrior', 'student_full_prior')
        logistic_names = ('FoELogisticFullPrior', 'FoELogisticFullPrior', 'logistic_full_prior')
        dist_names = student_names if dist == 'student' else logistic_names
        name = name or dist_names[0]
        load_name = load_name or dist_names[1]
        dir_name = dir_name or dist_names[2]
        load_tensor_names = load_tensor_names or tensor_names

        return name, load_name, dir_name, load_tensor_names

    def mrf_loss(self, xw, ica_a_flat):
        if self.dist == 'logistic':
            return logistic_full_mrf_loss(xw, ica_a_flat)
        elif self.dist == 'student':
            return student_full_mrf_loss(xw, ica_a_flat)
        else:
            raise NotImplementedError

    def make_normed_filters(self, trainable, squeeze_alpha, add_to_var_list=True):
        ica_a = tf.get_variable('ica_a', shape=[self.n_components, 1], trainable=trainable, dtype=tf.float32,
                                initializer=tf.random_normal_initializer())
        ica_w = tf.get_variable('ica_w', shape=[self.n_features_white, self.n_components],
                                trainable=trainable, dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=0.001))
        if add_to_var_list:
            self.var_list.extend([ica_a, ica_w])

        w_norm = tf.norm(ica_w, ord=2, axis=0)
        ica_w = ica_w / w_norm

        if squeeze_alpha:
            ica_a = tf.squeeze(ica_a)

        extra_op = None
        extra_vals = None
        return ica_a, ica_w, extra_op, extra_vals

    def build(self, scope_suffix='', featmap_tensor=None):

        with tf.variable_scope(self.name):

            n_features_raw = self.ph * self.pw * self.n_channels
            whitening_tensor = tf.get_variable('whiten_mat', shape=[self.n_features_white, n_features_raw],
                                               dtype=tf.float32, trainable=False)

            ica_a, ica_w, _, _ = self.make_normed_filters(trainable=False, squeeze_alpha=True)

            whitened_mixing = tf.matmul(whitening_tensor, ica_w, transpose_a=True)
            featmap = self.get_tensors() if featmap_tensor is None else featmap_tensor

            if self.mean_mode in ('gc', 'gf') and self.sdev_mode in ('gc', 'gf'):
                normed_featmap = self.norm_feat_map_directly(featmap)  # shape [1, h, w, n_channels]

                whitened_mixing = tf.reshape(whitened_mixing, shape=[self.n_channels, self.ph,
                                                                     self.pw, self.n_components])
                whitened_mixing = tf.transpose(whitened_mixing, perm=[1, 2, 0, 3])

                xw = tf.nn.conv2d(normed_featmap, whitened_mixing, strides=[1, 1, 1, 1], padding='VALID')
                xw = tf.reshape(xw, shape=[-1, self.n_components])

            else:
                normed_patches = self.shape_and_norm_featmap(featmap)
                n_patches = normed_patches.get_shape()[0].value
                normed_patches = tf.reshape(normed_patches, shape=[n_patches, n_features_raw])
                xw = tf.matmul(normed_patches, whitened_mixing)

            self.loss = self.mrf_loss(xw, ica_a)
            self.var_list.append(whitening_tensor)

    def score_matching_loss(self, x_mat, ica_w, ica_a):
        if self.dist == 'logistic':
            return logistic_full_score_matching_loss(x_mat, ica_w, ica_a)
        if self.dist == 'student':
            return student_full_score_matching_loss(x_mat, ica_w, ica_a)

    def make_data_dir(self):
        d_str = str(self.ph) + 'x' + str(self.pw)
        if isinstance(self.sdev_mode, float):
            mode_str = '_mean_{0}_sdev_rescaled_{1}'.format(self.mean_mode, self.sdev_mode)
        else:
            mode_str = '_mean_{0}_sdev_{1}'.format(self.mean_mode, self.sdev_mode)

        if 'pre_img' in self.in_tensor_names or 'rgb_scaled' in self.in_tensor_names:
            subdir = 'image/' + d_str
        else:
            t_str = self.in_tensor_names[:-len(':0')].replace('/', '_')
            subdir = self.classifier + '/' + t_str + '_' + d_str + '_' + str(self.n_features_white) + 'feats'
        data_dir = '../data/patches/' + subdir + mode_str + '/'
        return data_dir

    def get_x_placeholder(self, batch_size):
        return tf.placeholder(dtype=tf.float32, shape=[batch_size, self.n_features_white], name='x_pl')

    def score_matching_graph(self, batch_size):
        ica_a, ica_w, extra_op, _ = self.make_normed_filters(trainable=True, squeeze_alpha=False)

        x_pl = self.get_x_placeholder(batch_size)
        loss, term_1, term_2 = self.score_matching_loss(x_mat=x_pl, ica_w=ica_w, ica_a=ica_a)
        return loss, term_1, term_2, x_pl, ica_a, ica_w, extra_op

    def train_prior(self, batch_size, n_iterations, lr=3.0e-6, lr_lower_points=(), grad_clip=100.0,
                    n_data_samples=100000, n_val_samples=500,
                    log_freq=5000, summary_freq=10, print_freq=100, test_freq=100,
                    prev_ckpt=0, optimizer_name='adam',
                    plot_filters=False, n_vis=144, stop_on_overfit=True):

        if not os.path.exists(self.load_path):
            os.makedirs(self.load_path)

        data_dir = self.make_data_dir()
        data_gen = self.patch_batch_gen(batch_size, data_dir=data_dir, n_samples=n_data_samples, data_mode='train')
        val_gen = self.patch_batch_gen(batch_size, data_dir=data_dir, n_samples=n_val_samples, data_mode='validate')

        with tf.Graph().as_default():
            with tf.variable_scope(self.name):
                self.add_preprocessing_to_graph(data_dir, self.whiten_mode)

                loss, term_1, term_2, x_pl, ica_a, ica_w, extra_op = self.score_matching_graph(batch_size)

                train_op, lr_pl = self.get_train_op(loss, optimizer_name, grad_clip)

                if self.load_name != self.name and prev_ckpt:
                    to_load = self.tensor_load_dict_by_name(tf.global_variables())
                    saver = tf.train.Saver(var_list=to_load)
                else:
                    saver = tf.train.Saver()

                checkpoint_file = os.path.join(self.load_path, 'ckpt')

                tf.summary.scalar('total_loss', loss)
                tf.summary.scalar('term_1', term_1)
                tf.summary.scalar('term_2', term_2)
                train_summary_op = tf.summary.merge_all()
                summary_writer = tf.summary.FileWriter(self.load_path + '/summaries')

                val_loss = tf.placeholder(dtype=tf.float32, shape=[], name='val_loss')
                val_summary_op = tf.summary.scalar('validation_loss', val_loss)

                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())

                    if prev_ckpt:
                        self.load_weights(sess, prev_ckpt)

                    start_time = time.time()
                    train_time = 0

                    for count in range(prev_ckpt + 1, prev_ckpt + n_iterations + 1):
                        data = next(data_gen)

                        if lr_lower_points and lr_lower_points[0][0] <= count:
                            lr = lr_lower_points[0][1]
                            print('new learning rate: ', lr)
                            lr_lower_points = lr_lower_points[1:]

                        batch_start = time.time()
                        batch_loss, _, summary_string = sess.run(fetches=[loss, train_op, train_summary_op],
                                                                 feed_dict={x_pl: data, lr_pl: lr})

                        train_time += time.time() - batch_start

                        if count % summary_freq == 0:
                            summary_writer.add_summary(summary_string, count)

                        if count % print_freq == 0:
                            print(batch_loss)

                        if count % (print_freq * 10) == 0:
                            self.print_verbose(data, x_pl, ica_a, ica_w, term_1, term_2, sess,
                                               count, n_iterations, prev_ckpt, train_time, start_time)

                        if test_freq > 0 and count % test_freq == 0:
                            self.run_validation(val_gen, x_pl, loss, val_loss, val_summary_op, summary_writer, sess,
                                                n_val_samples, batch_size, count, start_time, saver, checkpoint_file)
                            if stop_on_overfit and self.decide_break(test_freq, log_freq):
                                break

                        if count % log_freq == 0:
                            if extra_op is not None:
                                sess.run(extra_op)
                            saver.save(sess, checkpoint_file, write_meta_graph=False, global_step=count)

                    saver.save(sess, checkpoint_file, write_meta_graph=False, global_step=n_iterations + prev_ckpt)

                    if plot_filters:
                        unwhiten_mat = np.load(data_dir + 'unwhiten_' + self.whiten_mode + '.npy').astype(np.float32)
                        w_res = sess.run(ica_w)
                        self.plot_filters_after_training(w_res, unwhiten_mat, n_vis)

    @staticmethod
    def get_train_op(loss, optimizer_name, grad_clip):
        lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name='lr')
        opt = get_optimizer(name=optimizer_name, lr_pl=lr_pl)
        tvars = tf.trainable_variables()
        grads = tf.gradients(loss, tvars)
        tg_pairs = [k for k in zip(grads, tvars) if k[0] is not None]
        tg_clipped = [(tf.clip_by_value(k[0], -grad_clip, grad_clip), k[1])
                      for k in tg_pairs]
        opt_op = opt.apply_gradients(tg_clipped)
        return opt_op, lr_pl

    @staticmethod
    def print_verbose(data, x_pl, ica_a, ica_w, term_1, term_2, sess,
                      count, n_iterations, prev_ckpt, train_time, start_time):
        w_res, alp, t1, t2 = sess.run([ica_w, ica_a, term_1, term_2], feed_dict={x_pl: data})
        print('it: ', count, ' / ', n_iterations + prev_ckpt)
        print('max a: ', np.max(alp), ' min a: ', np.min(alp), ' mean a: ', np.mean(alp))
        print('max w: ', np.max(w_res), ' min w: ', np.min(w_res),
              'mean abs w: ', np.mean(np.abs(w_res)))
        print('term_1: ', t1, ' term_2: ', t2)

        train_ratio = 100.0 * train_time / (time.time() - start_time)
        print('{0:2.1f}% of the time spent in run calls'.format(train_ratio))

    def run_validation(self, val_gen, x_pl, loss, val_loss, val_summary_op, summary_writer, sess,
                       n_val_samples, batch_size, count, start_time, saver, checkpoint_file):
        val_loss_acc = 0.0
        num_runs = n_val_samples // batch_size
        for val_count in range(num_runs):
            val_feed_dict = {x_pl: next(val_gen)}
            val_batch_loss = sess.run(loss, feed_dict=val_feed_dict)
            val_loss_acc += val_batch_loss
        val_loss_acc /= num_runs
        val_summary_string = sess.run(val_summary_op, feed_dict={val_loss: val_loss_acc})
        summary_writer.add_summary(val_summary_string, count)
        print(('Iteration: {0:6d} Validation Error: {1:9.2f} ' +
               'Time: {2:5.1f} min').format(count, val_loss_acc,
                                            (time.time() - start_time) / 60))

        if self.min_val_loss < val_loss_acc:
            print('Validation loss increased! Risk of overfitting')
            if self.increasing_val_flag == 0:
                saver.save(sess, checkpoint_file, write_meta_graph=False, global_step=count)
            self.increasing_val_flag += 1
        else:
            self.increasing_val_flag = 0
            self.min_val_loss = val_loss_acc

    def decide_break(self, test_freq, log_freq):
        return self.increasing_val_flag >= log_freq // test_freq

    def plot_filters_after_training(self, w_res, unwhiten_mat, n_vis):
        comps = np.dot(w_res.T, unwhiten_mat)
        comps -= np.min(comps)
        comps /= np.max(comps)
        co = np.reshape(comps[:n_vis, :], [n_vis, 3, self.ph, self.pw])
        co = np.transpose(co, axes=[0, 2, 3, 1])
        plot_img_mats(co, color=True, rescale=True)

    def load_filters_for_plotting(self):
        with tf.Graph().as_default():

            with tf.variable_scope(self.name):
                n_features_raw = self.ph * self.pw * self.n_channels
                unwhitening_tensor = tf.get_variable('unwhiten_mat', shape=[self.n_features_white, n_features_raw],
                                                     dtype=tf.float32, trainable=False)
                self.var_list = [unwhitening_tensor]
                ica_a, ica_w, _, _ = self.make_normed_filters(trainable=False, squeeze_alpha=True)

            with tf.Session() as sess:
                self.load_weights(sess)
                unwhitening_mat, a_mat, w_mat = sess.run([unwhitening_tensor, ica_a, ica_w])

        return unwhitening_mat, a_mat, w_mat

    def plot_channels_all_filters(self, channel_ids, save_path, save_as_mat=False, save_as_plot=True, n_vis=144):
        """
        visualizes all filters of selected channels and saves these as one plot per channel.
        :param channel_ids: ...
        :param save_path: location to save plots
        :param save_as_mat: if true, saves each filter as channel x height x width matrix
        :param save_as_plot:  if true, saves each filter as image
        :param n_vis: ...
        :return: None
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        unwhitening_mat, a_mat, w_mat = self.load_filters_for_plotting()
        rotated_w_mat = np.dot(w_mat[:, :n_vis].T, unwhitening_mat)

        for channel_id in channel_ids:
            chan_start = self.ph * self.pw * channel_id
            chan_end = self.ph * self.pw * (channel_id + 1)
            flat_channel = rotated_w_mat[:, chan_start:chan_end]
            plottable_channels = np.reshape(flat_channel, [flat_channel.shape[0], self.ph, self.pw])

            if save_as_mat:
                file_name = 'channel_{}.npy'.format(channel_id)
                np.save(save_path + file_name, plottable_channels)
            if save_as_plot:
                file_name = 'channel_{}.png'.format(channel_id)

                plottable_channels -= np.min(plottable_channels)
                plottable_channels /= np.max(plottable_channels)
                plot_img_mats(plottable_channels, rescale=False, show=False, save_path=save_path + file_name)
                print('filter {} done'.format(channel_id))

    def plot_channels_top_filters(self, channel_ids, save_path, save_as_mat=False, save_as_plot=True, n_vis=144):
        """
        visualizes filters of selected channels sorted descending by norm of the filter
        saves these as one plot per channel.

        :param channel_ids: collection of filter indices
        :param save_path: location to save plots
        :param save_as_mat: if true, saves each filter as channel x height x width matrix
        :param save_as_plot:  if true, saves each filter as image
        :param n_vis: ...
        :return: None
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        unwhitening_mat, a_mat, w_mat = self.load_filters_for_plotting()
        rotated_w_mat = np.dot(w_mat.T, unwhitening_mat)

        for channel_id in channel_ids:
            chan_start = self.ph * self.pw * channel_id
            chan_end = self.ph * self.pw * (channel_id + 1)
            flat_channel = rotated_w_mat[:, chan_start:chan_end]

            w_min = np.min(flat_channel)
            w_max = np.max(flat_channel)

            w_norms = np.linalg.norm(flat_channel, axis=1)
            norm_ids = np.argsort(w_norms)[-n_vis:][::-1]

            flat_channel = flat_channel[norm_ids, :]
            print(np.linalg.norm(flat_channel, axis=1))
            plottable_channels = np.reshape(flat_channel, [flat_channel.shape[0], self.ph, self.pw])

            if save_as_mat:
                file_name = 'channel_{}.npy'.format(channel_id)
                np.save(save_path + file_name, plottable_channels)
            if save_as_plot:
                file_name = 'channel_{}.png'.format(channel_id)

                plottable_channels -= w_min
                plottable_channels /= (w_max - w_min)
                plot_img_mats(plottable_channels, rescale=False, show=False, save_path=save_path + file_name)
                print('filter {} done'.format(channel_id))
    
    def plot_filters_all_channels(self, filter_ids, save_path, save_as_mat=False, save_as_plot=True):
        """
        visualizes the patch for each channel of a trained filter and saves this as one plot.
        does so for the filter of each given index

        :param filter_ids: collection of filter indices
        :param save_path: location to save plots
        :param save_as_mat: if true, saves each filter as channel x height x width matrix
        :param save_as_plot:  if true, saves each filter as image
        :return: None
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        unwhitening_mat, a_mat, w_mat = self.load_filters_for_plotting()
        rotated_w_mat = np.dot(w_mat[:, filter_ids].T, unwhitening_mat)

        for idx, filter_id in enumerate(filter_ids):
            flat_filter = rotated_w_mat[idx, :]
            alpha = a_mat[filter_id]
            plottable_filters = np.reshape(flat_filter, [self.n_channels, self.ph, self.pw])

            if save_as_mat:
                file_name = 'filter_{}_alpha_{:.3e}.npy'.format(filter_id, float(alpha))
                np.save(save_path + file_name, plottable_filters)
            if save_as_plot:
                file_name = 'filter_{}_alpha_{:.3e}.png'.format(filter_id, float(alpha))

                plottable_filters -= np.min(plottable_filters)
                plottable_filters /= np.max(plottable_filters)
                plot_img_mats(plottable_filters, rescale=False, show=False, save_path=save_path + file_name)
                print('filter {} done'.format(filter_id))

    def plot_filters_top_alphas(self, n_filters, save_path, save_as_mat=False, save_as_plot=True):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        unwhitening_mat, a_mat, w_mat = self.load_filters_for_plotting()
        top_alphas = np.argsort(a_mat)[::-1]
        filter_ids = top_alphas[:n_filters]
        rotated_w_mat = np.dot(w_mat[:, filter_ids].T, unwhitening_mat)

        for idx, filter_id in enumerate(filter_ids):
            flat_filter = rotated_w_mat[idx, :]
            alpha = a_mat[filter_id]
            plottable_filters = np.reshape(flat_filter, [self.n_channels, self.ph, self.pw])

            if save_as_mat:
                file_name = 'filter_{}_alpha_{:.3e}.npy'.format(filter_id, float(alpha))
                np.save(save_path + file_name, plottable_filters)
            if save_as_plot:
                file_name = 'filter_{}_alpha_{:.3e}.png'.format(filter_id, float(alpha))

                plottable_filters -= np.min(plottable_filters)
                plottable_filters /= np.max(plottable_filters)
                plot_img_mats(plottable_filters, rescale=False, show=False, save_path=save_path + file_name)
                print('filter {} done'.format(filter_id))

    @staticmethod
    def get_load_path(dir_name, classifier, tensor_name, filter_dims, n_components,
                      n_features_white, mean_mode, sdev_mode, whiten_mode):
        d_str = str(filter_dims[0]) + 'x' + str(filter_dims[1])
        if 'img' in tensor_name or 'image' in tensor_name or 'rgb_scaled' in tensor_name:
            subdir = 'image/color'
        else:
            if tensor_name.lower().startswith('split'):
                tensor_name = ''.join(tensor_name.split('/')[1:])
            tensor_name = tensor_name[:-len(':0')].replace('/', '_')
            subdir = classifier + '/' + tensor_name

        cf_str = str(n_components) + 'comps_' + str(n_features_white) + 'feats'
        target_dir = '_' + d_str + '_' + cf_str
        if isinstance(sdev_mode, float):
            mode_str = '_mean_' + mean_mode + '_sdev_rescaled' + 'whiten_' + whiten_mode
        else:
            mode_str = '_mean_' + mean_mode + '_sdev_' + sdev_mode + '_whiten_' + whiten_mode
        load_path = '../logs/foe_priors/' + dir_name + '/' + subdir + target_dir + mode_str + '/'

        return load_path

    def add_preprocessing_to_graph(self, data_dir, whiten_mode):
        unwhiten_mat = np.load(data_dir + 'unwhiten_' + whiten_mode + '.npy').astype(np.float32)
        whiten_mat = np.load(data_dir + 'whiten_' + whiten_mode + '.npy').astype(np.float32)

        tf.get_variable(name='whiten_mat', initializer=whiten_mat, trainable=False, dtype=tf.float32)
        tf.get_variable(name='unwhiten_mat', initializer=unwhiten_mat, trainable=False, dtype=tf.float32)

        retrieve_modes = ('gc', 'gf')
        if self.mean_mode in retrieve_modes:
            mean_mat = np.load(data_dir + 'data_mean.npy').astype(np.float32)
            tf.get_variable('centering_mean', initializer=mean_mat, trainable=False, dtype=tf.float32)

        if self.sdev_mode in retrieve_modes:
            sdev_mat = np.load(data_dir + 'data_sdev.npy').astype(np.float32)
            tf.get_variable('rescaling_sdev', initializer=sdev_mat, trainable=False, dtype=tf.float32)

    def shape_and_norm_featmap(self, featmap):
        # tensor = featmap
        scaled_featmap = featmap * self.input_scaling

        filter_mat = flattening_filter((self.ph, self.pw, 1))

        flat_filter = tf.constant(filter_mat, dtype=tf.float32)
        x_pad = ((self.ph - 1) // 2, int(np.ceil((self.ph - 1) / 2)))
        y_pad = ((self.pw - 1) // 2, int(np.ceil((self.pw - 1) / 2)))
        conv_input = tf.pad(scaled_featmap, paddings=[(0, 0), x_pad, y_pad, (0, 0)], mode='REFLECT')

        feat_map_list = tf.split(conv_input, num_or_size_splits=conv_input.get_shape()[3].value, axis=3)

        flat_patches_list = [tf.nn.conv2d(k, flat_filter, strides=[1, 1, 1, 1], padding='VALID')
                             for k in feat_map_list]  # first, filter per channel
        flat_patches = tf.stack(flat_patches_list, axis=4)
        fps = [k.value for k in flat_patches.get_shape()]  # shape = [bs, h, w, n_fpc, n_c]
        assert fps[0] == 1  # commit to singular batch size

        n_patches = fps[1] * fps[2]
        # n_features_raw = fps[3] * fps[4]
        flat_patches = tf.reshape(flat_patches, shape=[fps[0], n_patches, fps[3], fps[4]])  # flatten h,w

        flat_patches = tf.squeeze(flat_patches)

        normed_patches, mean_sdev_list = preprocess_patch_tensor(flat_patches, self.mean_mode, self.sdev_mode)
        self.var_list.extend(mean_sdev_list)
        return normed_patches

    def norm_feat_map_directly(self, featmap):
        """
        if both mean and sdev modes use training set values (gc or gf),
        norming can be applied directly to the feature map (shape [bs, h, w, c])

        :return:
        """
        assert self.mean_mode in ('gc', 'gf') and self.sdev_mode in ('gc', 'gf')
        featmap = featmap * tf.constant(self.input_scaling, dtype=tf.float32)
        assert featmap.get_shape()[0].value == 1
        featmap = tf.squeeze(featmap)
        normed_featmap, mean_sdev_list = preprocess_featmap_tensor(featmap, self.mean_mode, self.sdev_mode)
        normed_featmap = tf.expand_dims(normed_featmap, axis=0)
        self.var_list.extend(mean_sdev_list)
        return normed_featmap

    def patch_batch_gen(self, batch_size, data_dir, n_samples, data_mode='train'):
        assert data_mode in ('train', 'validate')

        if data_mode == 'train':
            data_mat = np.memmap(data_dir + 'data_mat_' + self.whiten_mode + '_whitened.npy',
                                 dtype=np.float32, mode='r', shape=(n_samples, self.n_features_white))

            idx = 0
            while True:
                if idx + batch_size < n_samples:
                    batch = data_mat[idx:(idx + batch_size), :]
                    idx += batch_size
                else:
                    last_bit = data_mat[idx:, :]
                    idx = (idx + batch_size) % n_samples
                    first_bit = data_mat[:idx, :]
                    batch = np.concatenate((last_bit, first_bit), axis=0)
                yield batch
        else:
            data_mat = np.load(data_dir + 'val_mat.npy')
            assert data_mat.shape[0] == n_samples, 'expected {}, found {}'.format(n_samples, data_mat.shape[0])
            assert n_samples % batch_size == 0

            idx = 0
            while True:
                batch = data_mat[idx:(idx + batch_size), :]
                idx += batch_size
                idx = idx % n_samples
                yield batch

    def forward_opt_sgd(self, input_featmap, learning_rate, n_iterations, make_switch=True):

        def cond(*args):
            return tf.not_equal(args[0], tf.constant(n_iterations, dtype=tf.int32))

        def body(count, featmap):
            count += 1
            self.build(featmap_tensor=featmap)
            featmap_grad = tf.gradients(ys=self.get_loss(), xs=featmap)[0]
            featmap -= learning_rate * featmap_grad
            return count, featmap

        count_init = tf.constant(0, dtype=tf.int32)
        _, final_featmap = tf.while_loop(cond=cond, body=body, loop_vars=[count_init, input_featmap])

        if make_switch:
            activate = tf.placeholder(dtype=tf.bool)
            output = tf.cond(activate, lambda: final_featmap, lambda: input_featmap)
            return output, activate
        else:
            return final_featmap, None

    def forward_opt_adam(self, input_featmap, learning_rate, n_iterations, in_to_loss_featmap_fun=None,
                         make_switch=False, beta1=0.9, beta2=0.999, eps=1e-8, explicit_notation=False):

        def apply_adam(variable, gradients, m_acc, v_acc, iteration):
            beta1_tsr = tf.constant(beta1, dtype=tf.float32)
            beta2_tsr = tf.constant(beta2, dtype=tf.float32)
            eps_tsr = tf.constant(eps, dtype=tf.float32)
            m_new = beta1_tsr * m_acc + (1.0 - beta1_tsr) * gradients
            v_new = beta2_tsr * v_acc + (1.0 - beta2_tsr) * (gradients ** 2)
            beta1_t_term = (1.0 - (beta1_tsr ** iteration))
            beta2_t_term = (1.0 - (beta2_tsr ** iteration))

            if explicit_notation:  # unoptimized form, with epsilon as given in the paper
                m_hat = m_new / beta1_t_term
                v_hat = v_new / beta2_t_term
                variable -= learning_rate * m_hat / (tf.sqrt(v_hat) + eps_tsr)
            else:  # optimized and w/ different epsilon (hat): this mimics the behaviour of the tf.AdamOptimizer
                lr_mod = tf.sqrt(beta2_t_term) / beta1_t_term
                variable -= learning_rate * lr_mod * m_new / (tf.sqrt(v_new) + eps_tsr)

            return variable, m_new, v_new

        def cond(*args):
            return tf.not_equal(args[0], tf.constant(n_iterations, dtype=tf.float32))

        def body(count, featmap, m_acc, v_acc):
            count += 1
            if in_to_loss_featmap_fun is None:
                self.build(featmap_tensor=featmap)
            else:
                self.build(featmap_tensor=in_to_loss_featmap_fun(featmap))
            featmap_grad = tf.gradients(ys=self.get_loss(), xs=featmap)[0]
            featmap, m_acc, v_acc = apply_adam(featmap, featmap_grad, m_acc, v_acc, count)
            return count, featmap, m_acc, v_acc

        featmap_shape = [k.value for k in input_featmap.get_shape()]
        m_init = tf.constant(np.zeros(featmap_shape), dtype=tf.float32)
        v_init = tf.constant(np.zeros(featmap_shape), dtype=tf.float32)
        count_init = tf.constant(0, dtype=tf.float32)

        _, final_featmap, _, _ = tf.while_loop(cond=cond, body=body,
                                               loop_vars=[count_init, input_featmap, m_init, v_init])
        if make_switch:
            activate = tf.placeholder(dtype=tf.bool)
            output = tf.cond(activate, lambda: final_featmap, lambda: input_featmap)
            return output, activate
        else:
            return final_featmap, None
