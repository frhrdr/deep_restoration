import tensorflow as tf
import numpy as np
import time
import os
from modules.core_modules import LearnedPriorLoss
from utils.temp_utils import flattening_filter, plot_img_mats, get_optimizer
from utils.preprocessing import preprocess_patch_tensor, make_data_dir, preprocess_featmap_tensor
from utils.patch_prior_losses import logistic_full_mrf_loss, logistic_full_score_matching_loss, \
    student_full_mrf_loss, student_full_score_matching_loss
# _mean_lf_sdev_none


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

    @staticmethod
    def assign_names(dist, name, load_name, dir_name, load_tensor_names, tensor_names):
        dist_options = ('student', 'logistic')
        assert dist in dist_options

        student_names = ('FoEStudentFullPrior', 'FoEStudentFullPrior', 'student_full_prior')
        logistic_names = ('FoELogisticFullPrior', 'FoELogisticFullPrior', 'logistic_full_prior')
        dist_names = student_names if dist == 'student' else logistic_names
        name = name if name is not None else dist_names[0]
        load_name = load_name if load_name is not None else dist_names[1]
        dir_name = dir_name if dir_name is not None else dist_names[2]
        load_tensor_names = load_tensor_names if load_tensor_names is not None else tensor_names

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
        ica_w = ica_w / tf.norm(ica_w, ord=2, axis=0)
        if squeeze_alpha:
            ica_a = tf.squeeze(ica_a)

        return ica_a, ica_w

    def build(self, scope_suffix=''):

        with tf.variable_scope(self.name):

            n_features_raw = self.ph * self.pw * self.n_channels
            whitening_tensor = tf.get_variable('whiten_mat', shape=[self.n_features_white, n_features_raw],
                                               dtype=tf.float32, trainable=False)

            ica_a, ica_w = self.make_normed_filters(trainable=False, squeeze_alpha=True)

            whitened_mixing = tf.matmul(whitening_tensor, ica_w, transpose_a=True)

            if self.mean_mode in ('gc', 'gf') and self.sdev_mode in ('gc', 'gf'):
                normed_featmap = self.norm_feat_map_directly()  # shape [1, h, w, n_channels]

                whitened_mixing = tf.reshape(whitened_mixing, shape=[self.n_channels, self.ph,
                                                                     self.pw, self.n_components])
                whitened_mixing = tf.transpose(whitened_mixing, perm=[1, 2, 0, 3])
                print(normed_featmap.get_shape())
                print(whitened_mixing.get_shape())
                xw = tf.nn.conv2d(normed_featmap, whitened_mixing, strides=[1, 1, 1, 1], padding='VALID')
                xw = tf.reshape(xw, shape=[-1, self.n_components])

            else:
                normed_patches = self.shape_and_norm_tensor()
                n_patches = normed_patches.get_shape()[0].value
                normed_patches = tf.reshape(normed_patches, shape=[n_patches, n_features_raw])
                xw = tf.matmul(normed_patches, whitened_mixing)

            self.loss = self.mrf_loss(xw, ica_a)
            self.var_list.extend([ica_a, ica_w, whitening_tensor])

    def score_matching_loss(self, x_mat, ica_w, ica_a):
        if self.dist == 'logistic':
            return logistic_full_score_matching_loss(x_mat, ica_w, ica_a)
        if self.dist == 'student':
            return student_full_score_matching_loss(x_mat, ica_w, ica_a)

    def train_prior(self, batch_size, n_iterations, lr=3.0e-6, lr_lower_points=(), grad_clip=100.0, n_vis=144,
                    n_data_samples=100000, n_val_samples=500,
                    log_freq=5000, summary_freq=10, print_freq=100, test_freq=100,
                    prev_ckpt=0, optimizer_name='adam',
                    plot_filters=False, do_clip=True):
        log_path = self.load_path

        data_dir = make_data_dir(in_tensor_name=self.in_tensor_names, ph=self.ph, pw=self.pw,
                                 mean_mode=self.mean_mode, sdev_mode=self.sdev_mode,
                                 n_features_white=self.n_features_white, classifier=self.classifier)

        data_gen = self.patch_batch_gen(batch_size, data_dir=data_dir, n_samples=n_data_samples, data_mode='train')

        val_gen = self.patch_batch_gen(batch_size, data_dir=data_dir, n_samples=n_data_samples, data_mode='validate')

        with tf.Graph().as_default() as graph:
            with tf.variable_scope(self.name):
                self.add_preprocessing_to_graph(data_dir, self.whiten_mode)

                x_pl = tf.placeholder(dtype=tf.float32, shape=[batch_size, self.n_features_white], name='x_pl')
                lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name='lr')

                ica_a, ica_w = self.make_normed_filters(trainable=True, squeeze_alpha=False)
                loss, term_1, term_2 = self.score_matching_loss(x_mat=x_pl, ica_w=ica_w, ica_a=ica_a)

                opt = get_optimizer(name=optimizer_name, lr_pl=lr_pl)
                tvars = tf.trainable_variables()
                grads = tf.gradients(loss, tvars)
                tg_pairs = [k for k in zip(grads, tvars) if k[0] is not None]
                tg_clipped = [(tf.clip_by_value(k[0], -grad_clip, grad_clip), k[1])
                              for k in tg_pairs]
                opt_op = opt.apply_gradients(tg_clipped)

                if self.load_name != self.name and prev_ckpt:
                    # names = [k.name.split('/') for k in tf.global_variables()]
                    # prev_load_name = names[0][0]
                    # names = [[l if l != prev_load_name else self.load_name for l in k] for k in names]
                    # names = ['/'.join(k) for k in names]
                    # names = [k.split(':')[0] for k in names]
                    # to_load = dict(zip(names, tf.global_variables()))
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

                val_loss = tf.placeholder(dtype=tf.float32, shape=[], name='val_loss')
                val_summary_op = tf.summary.scalar('validation_loss', val_loss)

                with tf.Session() as sess:
                    # print(tf.trainable_variables())
                    sess.run(tf.global_variables_initializer())

                    if prev_ckpt:
                        prev_path = self.load_path + 'ckpt-' + str(prev_ckpt)
                        saver.restore(sess, prev_path)

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

                        if test_freq > 0 and count % test_freq == 0:
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

                        if count % log_freq == 0:
                            saver.save(sess, checkpoint_file, write_meta_graph=False, global_step=count)

                    saver.save(sess, checkpoint_file, write_meta_graph=False, global_step=n_iterations + prev_ckpt)

                    if plot_filters:
                        unwhiten_mat = np.load(data_dir + 'unwhiten_' + self.whiten_mode + '.npy').astype(np.float32)
                        w_res, alp = sess.run([ica_w, ica_a])
                        comps = np.dot(w_res.T, unwhiten_mat)
                        # print(comps.shape)
                        comps -= np.min(comps)
                        comps /= np.max(comps)
                        # co = np.reshape(comps[:n_vis, :], [n_vis, ph, pw, 3])
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
                ica_a, ica_w = self.make_normed_filters(trainable=False, squeeze_alpha=True)

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
        if 'img' in tensor_name or 'image' in tensor_name:
            subdir = 'image/color'
        else:
            if tensor_name.lower().startswith('split'):
                tensor_name = ''.join(tensor_name.split('/')[1:])
            tensor_name = tensor_name[:-len(':0')].replace('/', '_')
            subdir = classifier + '/' + tensor_name

        cf_str = str(n_components) + 'comps_' + str(n_features_white) + 'feats'
        target_dir = '_' + d_str + '_' + cf_str
        if isinstance(sdev_mode, float):
            mode_str = '_mean_' + mean_mode + '_sdev_rescaled'
        else:
            mode_str = '_mean_' + mean_mode + '_sdev_' + sdev_mode
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

    def shape_and_norm_tensor(self):
        tensor = self.get_tensors()
        scaled_tensor = tensor * self.input_scaling

        filter_mat = flattening_filter((self.ph, self.pw, 1))

        flat_filter = tf.constant(filter_mat, dtype=tf.float32)
        x_pad = ((self.ph - 1) // 2, int(np.ceil((self.ph - 1) / 2)))
        y_pad = ((self.pw - 1) // 2, int(np.ceil((self.pw - 1) / 2)))
        conv_input = tf.pad(scaled_tensor, paddings=[(0, 0), x_pad, y_pad, (0, 0)], mode='REFLECT')

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

    def norm_feat_map_directly(self):
        """
        if both mean and sdev modes use training set values (gc or gf),
        norming can be applied directly to the feature map (shape [bs, h, w, c])

        :return:
        """
        assert self.mean_mode in ('gc', 'gf') and self.sdev_mode in ('gc', 'gf')

        featmap = self.get_tensors()
        featmap = featmap * self.input_scaling
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
