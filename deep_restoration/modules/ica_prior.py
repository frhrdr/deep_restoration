import tensorflow as tf
import numpy as np
from modules.loss_modules import LearnedPriorLoss
from utils.temp_utils import flattening_filter, patch_batch_gen, plot_img_mats, get_optimizer
from utils.preprocessing import preprocess_tensor
import time
import os
# _mean_lf_sdev_none

class ICAPrior(LearnedPriorLoss):

    def __init__(self, tensor_names, weighting, classifier, filter_dims, input_scaling, n_components, n_channels,
                 n_features_white,
                 trainable=False, name='ICAPrior',load_name='ICAPrior', dir_name='ica_prior',
                 mean_mode='lf', sdev_mode='none'):
        mode_abbreviatons = {'global_channel': 'gc', 'global_feature': 'gf', 'local_channel': 'lc', 'local_full': 'lf'}
        if mean_mode in mode_abbreviatons:
            mean_mode = mode_abbreviatons[mean_mode]
        if sdev_mode in mode_abbreviatons:
            sdev_mode = mode_abbreviatons[sdev_mode]

        load_path = self.get_load_path(dir_name, classifier, tensor_names, filter_dims,
                                       n_components, n_features_white, mean_mode, sdev_mode)

        super().__init__(tensor_names, weighting, name, load_path, trainable, load_name)
        self.filter_dims = filter_dims  # tuple (height, width)
        self.input_scaling = input_scaling  # most likely 1, 255 or 1/255
        self.n_components = n_components  # number of components to be produced
        self.n_channels = n_channels
        self.n_features_white = n_features_white
        self.dir_name = dir_name
        self.classifier = classifier
        self.mean_mode = mean_mode
        self.sdev_mode = sdev_mode


    def build(self, scope_suffix=''):
        with tf.variable_scope(self.name):
            tensor = self.get_tensors()
            scaled_tensor = tensor * self.input_scaling

            # filter_mat = flattening_filter((self.filter_dims[0], self.filter_dims[1], dims[3]))
            filter_mat = flattening_filter((self.filter_dims[0], self.filter_dims[1], 1))

            flat_filter = tf.constant(filter_mat, dtype=tf.float32)
            x_pad = ((self.filter_dims[0] - 1) // 2, int(np.ceil((self.filter_dims[0] - 1) / 2)))
            y_pad = ((self.filter_dims[1] - 1) // 2, int(np.ceil((self.filter_dims[1] - 1) / 2)))
            conv_input = tf.pad(scaled_tensor, paddings=[(0, 0), x_pad, y_pad, (0, 0)], mode='REFLECT')
            # flat_patches = tf.nn.conv2d(conv_input, flat_filter, strides=[1, 1, 1, 1], padding='VALID')

            feat_map_list = tf.split(conv_input, num_or_size_splits=conv_input.get_shape()[3].value, axis=3)
            flat_patches_list = [tf.nn.conv2d(k, flat_filter, strides=[1, 1, 1, 1], padding='VALID')
                                 for k in feat_map_list]  # first, filter per channel
            flat_patches = tf.stack(flat_patches_list, axis=4)
            fps = [k.value for k in flat_patches.get_shape()]  # shape = [bs, h, w, n_fpc, n_c]
            n_patches = fps[1] * fps[2]
            n_features_raw = fps[3] * fps[4]
            flat_patches = tf.reshape(flat_patches, shape=[fps[0], n_patches, fps[3], fps[4]])  # flatten h,w
            assert fps[0] == 1  # commit to singular batch size
            flat_patches = tf.squeeze(flat_patches)

            normed_patches = preprocess_tensor(flat_patches, self.mean_mode, self.sdev_mode)
            normed_patches = tf.reshape(normed_patches, shape=[n_patches, n_features_raw])
            # flat_patches = tf.reshape(flat_patches, shape=[n_patches, n_features_raw])
            # means = tf.stack([tf.reduce_mean(flat_patches, axis=1)] * n_features_raw, axis=1)
            # centered_patches = flat_patches - means

            whitening_tensor = tf.get_variable('whiten_mat', shape=[self.n_features_white, n_features_raw],
                                               dtype=tf.float32, trainable=False)

            ica_a = tf.get_variable('ica_a', shape=[self.n_components, 1], trainable=self.trainable, dtype=tf.float32)
            ica_w = tf.get_variable('ica_w', shape=[self.n_features_white, self.n_components],
                                    trainable=self.trainable, dtype=tf.float32)
            ica_a_squeezed = tf.squeeze(ica_a)
            whitened_mixing = tf.matmul(whitening_tensor, ica_w, transpose_a=True)

            normed_patches = tf.Print(normed_patches, [tf.reduce_mean(normed_patches),
                                                       tf.reduce_max(normed_patches), tf.reduce_max(flat_patches),
                                                       tf.reduce_max(tensor), tf.reduce_max(ica_a_squeezed)])
            normed_patches = tf.Print(normed_patches, [tf.reduce_max(whitened_mixing)])
            xw = tf.matmul(normed_patches, whitened_mixing)
            neg_g_wx = tf.log(0.5 * (tf.exp(-xw) + tf.exp(xw))) * ica_a_squeezed
            neg_log_p_patches = tf.reduce_sum(neg_g_wx, axis=1)
            naive_mean = tf.reduce_mean(neg_log_p_patches, name='loss')
            naive_mean = tf.Print(naive_mean, [tf.reduce_max(xw), tf.reduce_max(neg_log_p_patches),
                                               tf.reduce_max(neg_g_wx), tf.reduce_max(naive_mean)])
            self.loss = naive_mean

            self.var_list = [ica_a, ica_w, whitening_tensor]

    @staticmethod
    def score_matching_loss(x_mat, w_mat, alpha):
        const_t = x_mat.get_shape()[0].value
        xw_mat = tf.matmul(x_mat, w_mat)
        g_mat = -tf.tanh(xw_mat)
        gp_mat = -4.0 / tf.square(tf.exp(xw_mat) + tf.exp(-xw_mat))  # d/dx tanh(x) = 4 / (exp(x) + exp(-x))^2
        gp_vec = tf.reduce_sum(gp_mat, axis=0) / const_t
        gg_mat = tf.matmul(g_mat, g_mat, transpose_a=True) / const_t
        aa_mat = tf.matmul(alpha, alpha, transpose_b=True)
        ww_mat = tf.matmul(w_mat, w_mat, transpose_a=True)
        w_norm = tf.diag_part(ww_mat, name='w_norm')

        term_1 = tf.reduce_sum(alpha * w_norm * gp_vec, name='t1')
        term_2 = 0.5 * tf.reduce_sum(aa_mat * ww_mat * gg_mat, name='t2')
        return term_1 + term_2, term_1, term_2

    def train_prior(self, batch_size, num_iterations, lr=3.0e-6, lr_lower_points=(), grad_clip=100.0, n_vis=144,
                    whiten_mode='pca', num_data_samples=100000,
                    log_freq=5000, summary_freq=10, print_freq=100, prev_ckpt=0, optimizer_name='momentum',
                    plot_filters=False, do_clip=True):
        log_path = self.load_path
        ph, pw = self.filter_dims

        data_dir = self.make_data_dir()

        data_gen = patch_batch_gen(batch_size, whiten_mode=whiten_mode, data_dir=data_dir,
                                   data_shape=(num_data_samples, self.n_features_white))
        # unwhiten_mat = np.load(data_dir + 'unwhiten_' + whiten_mode + '.npy').astype(np.float32)
        # whiten_mat = np.load(data_dir + 'whiten_' + whiten_mode + '.npy').astype(np.float32)
        with tf.Graph().as_default() as graph:
            with tf.variable_scope(self.name):
                # add whitening mats to the save-files for later retrieval
                # tf.get_variable(name='whiten_mat', initializer=whiten_mat, trainable=False, dtype=tf.float32)
                # tf.get_variable(name='unwhiten_mat', initializer=unwhiten_mat, trainable=False, dtype=tf.float32)
                self.add_preprocessing_to_graph(data_dir, whiten_mode)

                x_pl = tf.placeholder(dtype=tf.float32, shape=[batch_size, self.n_features_white], name='x_pl')
                lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name='lr')

                w_mat = tf.get_variable(shape=[self.n_features_white, self.n_components], dtype=tf.float32, name='ica_w',
                                        initializer=tf.random_normal_initializer(stddev=0.001))
                alpha = tf.get_variable(shape=[self.n_components, 1], dtype=tf.float32, name='ica_a',
                                        initializer=tf.random_normal_initializer())

                loss, term_1, term_2 = self.score_matching_loss(x_mat=x_pl, w_mat=w_mat, alpha=alpha)

                clip_op = tf.assign(w_mat, w_mat / tf.norm(w_mat, ord=2, axis=0))

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

                        if do_clip:
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
                        comps = np.dot(w_res.T, unwhiten_mat)
                        print(comps.shape)
                        comps -= np.min(comps)
                        comps /= np.max(comps)
                        # co = np.reshape(comps[:n_vis, :], [n_vis, ph, pw, 3])
                        co = np.reshape(comps[:n_vis, :], [n_vis, 3, ph, pw])
                        co = np.transpose(co, axes=[0, 2, 3, 1])
                        plot_img_mats(co, color=True, rescale=True)

    def plot_filters(self, filter_ids, save_path, save_as_mat=False, save_as_plot=True):
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

        with tf.Graph().as_default():

            with tf.variable_scope(self.name):
                n_features_raw = self.filter_dims[0] * self.filter_dims[1] * self.n_channels
                unwhitening_tensor = tf.get_variable('unwhiten_mat', shape=[self.n_features_white, n_features_raw],
                                                     dtype=tf.float32, trainable=False)
                ica_a = tf.get_variable('ica_a', shape=[self.n_components, 1], trainable=self.trainable,
                                        dtype=tf.float32)
                ica_w = tf.get_variable('ica_w', shape=[self.n_features_white, self.n_components],
                                        trainable=self.trainable, dtype=tf.float32)
            self.var_list = [unwhitening_tensor, ica_a, ica_w]

            with tf.Session() as sess:
                self.load_weights(sess)
                unwhitening_mat, a_mat, w_mat = sess.run(self.var_list)

            print('matrices loaded')

            rotated_w_mat = np.dot(w_mat[:, filter_ids].T, unwhitening_mat)

            print('whitening reversed')

            for idx, filter_id in enumerate(filter_ids):
                flat_filter = rotated_w_mat[idx, :]
                alpha = a_mat[filter_id]
                chan_filter = np.reshape(flat_filter, [self.filter_dims[0], self.filter_dims[1], self.n_channels])
                plottable_filters = np.rollaxis(chan_filter, 2)

                if save_as_mat:
                    file_name = 'filter_{}_alpha_{:.3e}.npy'.format(filter_id, float(alpha))
                    np.save(save_path + file_name, plottable_filters)
                if save_as_plot:
                    file_name = 'filter_{}_alpha_{:.3e}.png'.format(filter_id, float(alpha))
                    plot_img_mats(plottable_filters, rescale=True, show=False, save_path=save_path + file_name)
                    print('filter {} done'.format(filter_id))

    @staticmethod
    def get_load_path(dir_name, classifier, tensor_name, filter_dims, n_components,
                      n_features_white, mean_mode, sdev_mode):
        d_str = str(filter_dims[0]) + 'x' + str(filter_dims[1])
        if 'pre_img' in tensor_name:
            subdir = 'image/color'
        else:
            subdir = classifier + '/' + tensor_name[:-len(':0')].replace('/', '_')

        cf_str = str(n_components) + 'comps_' + str(n_features_white) + 'feats'
        target_dir =  '_' + d_str + '_' + cf_str
        mode_str = '_mean_' + mean_mode + '_sdev_' + sdev_mode
        load_path = '../logs/priors/' + dir_name + '/' + subdir + target_dir + mode_str + '/'

        return load_path

    def make_data_dir(self):
        d_str = str(self.filter_dims[0]) + 'x' + str(self.filter_dims[1])
        mode_str = ''
        retrieve_modes = ('gc', 'gf')
        if self.mean_mode in retrieve_modes or self.sdev_mode in retrieve_modes:
            mode_str = '_mean_{0}_sdev_{1}'.format(self.mean_mode, self.sdev_mode)
        if 'pre_img' in self.in_tensor_names:
            subdir = 'image/' + d_str
        else:
            t_str = self.in_tensor_names[:-len(':0')].replace('/', '_')
            subdir = self.classifier + '/' + t_str + '_' + d_str + '_' + str(self.n_features_white) + 'feats'
        data_dir = '../data/patches/' + subdir + mode_str + '/'
        return data_dir


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
