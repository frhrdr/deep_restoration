import tensorflow as tf
import numpy as np
import time
import os
from utils.temp_utils import patch_batch_gen, plot_img_mats, get_optimizer
from utils.preprocessing import preprocess_patch_tensor, make_data_dir, preprocess_featmap_tensor
from modules.foe_full_prior import FoEFullPrior


class FoESeparablePrior(FoEFullPrior):

    def __init__(self, tensor_names, weighting, classifier, filter_dims, input_scaling, n_components, n_channels,
                 n_features_white, dim_multiplier, dist='logistic', mean_mode='gc', sdev_mode='gc',
                 trainable=False, name=None, load_name=None, dir_name=None, load_tensor_names=None):

        super().__init__(tensor_names, weighting, classifier, filter_dims, input_scaling, n_components, n_channels,
                         n_features_white, dist=dist, mean_mode=mean_mode, sdev_mode=sdev_mode, trainable=trainable,
                         name=name, load_name=load_name, dir_name=dir_name, load_tensor_names=load_tensor_names)

        self.dim_multiplier = dim_multiplier

    @staticmethod
    def assign_names(dist, name, load_name, dir_name, load_tensor_names, tensor_names):
        dist_options = ('student', 'logistic')
        assert dist in dist_options

        student_names = ('FoEStudentSeparablePrior', 'FoEStudentSeparablePrior', 'student_separable_prior')
        logistic_names = ('FoELogisticSeparablePrior', 'FoELogisticSeparablePrior', 'logistic_separable_prior')
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
        return path.rstrip('/') + '_separable/'

    def build(self, scope_suffix=''):

        with tf.variable_scope(self.name):

            n_features_per_channel = self.ph * self.pw
            n_features_raw = n_features_per_channel * self.n_channels
            whitening_tensor = tf.get_variable('whiten_mat', shape=[self.n_features_white, n_features_raw],
                                               dtype=tf.float32, trainable=False)

            ica_a = tf.get_variable('ica_a', shape=[self.n_components, 1], trainable=self.trainable, dtype=tf.float32)
            ica_a_squeezed = tf.squeeze(ica_a)

            depth_filter = tf.get_variable('depth_filter', shape=[self.ph, self.pw,
                                                                  self.n_channels, self.dim_multiplier])

            point_filter = tf.get_variable('point_filter', shape=[1, 1, self.n_channels * self.dim_multiplier,
                                                                  self.n_components])
            ica_w = None

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

            self.loss = self.mrf_loss(xw, ica_a_squeezed)
            self.var_list.extend([ica_a, ica_w, whitening_tensor])

    def train_prior(self, batch_size, n_iterations, lr=3.0e-6, lr_lower_points=(), grad_clip=100.0, n_vis=144,
                    whiten_mode='pca', n_data_samples=100000, n_val_samples=500,
                    log_freq=5000, summary_freq=10, print_freq=100, test_freq=100,
                    prev_ckpt=0, optimizer_name='adam',
                    plot_filters=False, do_clip=True):
        log_path = self.load_path

        data_dir = make_data_dir(in_tensor_name=self.in_tensor_names, ph=self.ph, pw=self.pw,
                                 mean_mode=self.mean_mode, sdev_mode=self.sdev_mode,
                                 n_features_white=self.n_features_white, classifier=self.classifier)

        data_gen = patch_batch_gen(batch_size, whiten_mode=whiten_mode, data_dir=data_dir,
                                   data_shape=(n_data_samples, self.n_features_white), data_mode='train')

        val_gen = patch_batch_gen(batch_size, whiten_mode=whiten_mode, data_dir=data_dir,
                                  data_shape=(n_val_samples, self.n_features_white), data_mode='validate')

        with tf.Graph().as_default() as graph:
            with tf.variable_scope(self.name):
                self.add_preprocessing_to_graph(data_dir, whiten_mode)

                x_pl = tf.placeholder(dtype=tf.float32, shape=[batch_size, self.n_features_white], name='x_pl')
                lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name='lr')

                depth_filter = tf.get_variable('depth_filter', shape=[self.ph, self.pw,
                                                                      self.n_channels, self.dim_multiplier])

                point_filter = tf.get_variable('point_filter', shape=[1, 1, self.n_channels * self.dim_multiplier,
                                                                      self.n_components])

                ica_a = tf.get_variable(shape=[self.n_components, 1], dtype=tf.float32, name='ica_a',
                                        initializer=tf.random_normal_initializer())

                ica_w = self.get_w_tensor(depth_filter, point_filter)

                loss, term_1, term_2 = self.score_matching_loss(x_mat=x_pl, ica_w=ica_w, ica_a=ica_a)

                flat_depth_filter = tf.reshape(depth_filter, shape=(self.n_features_white, self.dim_multiplier))
                flat_point_filter = tf.squeeze(point_filter)

                # clip_op = tf.assign(w_mat, w_mat / tf.norm(w_mat, ord=2, axis=0))
                # clip_op1 = tf.assign(depth_filter, depth_filter/ tf.norm(depth_filter, ord=2, axis=None))
                clip_op1 = tf.assign(depth_filter, depth_filter / tf.norm(flat_depth_filter, ord=2, axis=0))
                clip_op2 = tf.assign(point_filter, point_filter / tf.norm(flat_point_filter, ord=2, axis=0))

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

                val_loss = tf.placeholder(dtype=tf.float32, shape=[], name='val_loss')
                val_summary_op = tf.summary.scalar('validation_loss', val_loss)

                with tf.Session() as sess:
                    # print(tf.trainable_variables())
                    sess.run(tf.global_variables_initializer())

                    if prev_ckpt:
                        prev_path = self.load_path + 'ckpt-' + str(prev_ckpt)
                        saver.restore(sess, prev_path)

                    sess.run([clip_op1, clip_op2])

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

                        if do_clip:
                            sess.run([clip_op1, clip_op2])

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
                        unwhiten_mat = np.load(data_dir + 'unwhiten_' + whiten_mode + '.npy').astype(np.float32)
                        w_res, alp = sess.run([ica_w, ica_a])
                        comps = np.dot(w_res.T, unwhiten_mat)
                        # print(comps.shape)
                        comps -= np.min(comps)
                        comps /= np.max(comps)
                        # co = np.reshape(comps[:n_vis, :], [n_vis, ph, pw, 3])
                        co = np.reshape(comps[:n_vis, :], [n_vis, 3, self.ph, self.pw])
                        co = np.transpose(co, axes=[0, 2, 3, 1])
                        plot_img_mats(co, color=True, rescale=True)

    def get_w_tensor(self, depth_filter, point_filter):

        depth_filter = tf.reshape(depth_filter, shape=(self.ph * self.pw, self.n_channels * self.dim_multiplier, 1))
        depth_filter = tf.transpose(depth_filter, perm=(1, 0, 2))

        point_filter = tf.reshape(point_filter, shape=(self.n_channels * self.dim_multiplier, self.n_components, 1))
        point_filter = tf.transpose(point_filter, perm=(0, 2, 1))

        q_tensor = tf.matmul(depth_filter, point_filter, name='Q')
        q_tensor = tf.reshape(q_tensor, shape=(self.n_channels, self.dim_multiplier,
                                               self.ph * self.pw, self.n_components))

        w_tensor = tf.reduce_sum(q_tensor, axis=1, name='W')
        w_tensor = tf.reshape(w_tensor, shape=(self.n_channels * self.ph * self.pw, self.n_components))

        return w_tensor

    def validate_w_build(self, prev_ckpt):

        with tf.Graph().as_default() as graph:
            with tf.variable_scope(self.name):
                x_init = np.random.normal(size=(1, self.ph, self.pw, self.n_channels))
                x_patch = tf.constant(x_init, dtype=tf.float32, name='x_flat')
                x_flat = tf.reshape(tf.transpose(x_patch, perm=(0, 3, 1, 2)), shape=(1, self.n_features_white))

                depth_filter = tf.get_variable('depth_filter', shape=[self.ph, self.pw,
                                                                      self.n_channels, self.dim_multiplier])

                point_filter = tf.get_variable('point_filter', shape=[1, 1, self.n_channels * self.dim_multiplier,
                                                                      self.n_components])

                w_mat = self.get_w_tensor(depth_filter, point_filter)
                xw_mat_based = tf.squeeze(tf.matmul(x_flat, w_mat))

                xw_conv_based = tf.squeeze(tf.nn.separable_conv2d(x_patch, depth_filter, point_filter,
                                                                  strides=(1, 1, 1, 1), padding='VALID'))

                print('mat ', xw_mat_based.get_shape())
                print('conv', xw_conv_based.get_shape())

                if self.load_name != self.name and prev_ckpt:
                    to_load = self.tensor_load_dict_by_name(tf.global_variables())
                    saver = tf.train.Saver(var_list=to_load)
                else:
                    saver = tf.train.Saver()

                with tf.Session() as sess:
                    # print(tf.trainable_variables())
                    sess.run(tf.global_variables_initializer())

                    if prev_ckpt:
                        checkpoint_file = os.path.join(self.load_path, 'ckpt')
                        prev_path = self.load_path + 'ckpt-' + str(prev_ckpt)
                        saver.restore(sess, prev_path)


                    xw_mat, xw_conv = sess.run(fetches=[xw_mat_based, xw_conv_based])

                    print(xw_mat[:30])
                    print(xw_conv[:30])
                    print(xw_mat / xw_conv)
                    print(np.max(np.abs(xw_mat / xw_conv - 1)))

    def load_filters_for_plotting(self):
        with tf.Graph().as_default():

            with tf.variable_scope(self.name):
                n_features_raw = self.ph * self.pw * self.n_channels
                unwhitening_tensor = tf.get_variable('unwhiten_mat', shape=[self.n_features_white, n_features_raw],
                                                     dtype=tf.float32, trainable=False)
                depth_filter = tf.get_variable('depth_filter', shape=[self.ph, self.pw,
                                                                      self.n_channels, self.dim_multiplier])

                point_filter = tf.get_variable('point_filter', shape=[1, 1, self.n_channels * self.dim_multiplier,
                                                                      self.n_components])

                ica_a = tf.get_variable(shape=[self.n_components, 1], dtype=tf.float32, name='ica_a',
                                        initializer=tf.random_normal_initializer())

                ica_w = self.get_w_tensor(depth_filter, point_filter)
            self.var_list = [unwhitening_tensor, ica_a, depth_filter, point_filter]
            fetch_list = [unwhitening_tensor, ica_a, ica_w]

            with tf.Session() as sess:
                self.load_weights(sess)
                unwhitening_mat, a_mat, w_mat = sess.run(fetch_list)

            return unwhitening_mat, a_mat, w_mat
