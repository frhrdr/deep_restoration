import tensorflow as tf
from tf_vgg import vgg16
from tf_alexnet import alexnet
import filehandling_utils
from parameter_utils import check_params
from filehandling_utils import save_dict
from inv_modules import TrainedModule
from loss_modules import LossModule, LearnedPriorLoss
import os
import time
import numpy as np
from skimage.color import grey2rgb
import matplotlib
matplotlib.use('tkagg', force=True)
import matplotlib.pyplot as plt


class NetInversion:

    def __init__(self, params):
        check_params(params)
        self.params = params
        self.imagenet_mean = np.asarray([123.68, 116.779, 103.939])  # in RGB order
        self.img_hw = 224
        self.img_channels = 3

    def load_classifier(self, img_pl):
        if self.params['classifier'].lower() == 'vgg16':
            classifier = vgg16.Vgg16()
        elif self.params['classifier'].lower() == 'alexnet':
            classifier = alexnet.AlexNet()
        else:
            raise NotImplementedError

        classifier.build(img_pl, rescale=1.0)

    def build_model(self):

        if not isinstance(self.params['inv_modules'], list):
            self.params['inv_modules'] = [self.params['inv_modules']]

        for idx, inv_mod in enumerate(self.params['inv_modules']):
            assert isinstance(inv_mod, TrainedModule)
            inv_mod.build(scope_suffix=str(idx))
        loss = 0
        for loss_mod in self.params['loss_modules']:
            assert isinstance(loss_mod, LossModule)
            loss_mod.build()
            loss += loss_mod.get_loss()

        if self.params['optimizer'].lower() == 'adam':
            adam = tf.train.AdamOptimizer(learning_rate=self.params['learning_rate'])
            train_op = adam.minimize(loss)
        else:
            raise NotImplementedError

        return loss, train_op

    def build_logging(self, loss):
        tf.summary.scalar('total_loss', loss)
        for loss_mod in self.params['loss_modules']:
            loss_mod.scalar_summary()
        train_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.params['log_path'] + '/summaries')
        saver = tf.train.Saver()

        val_loss = tf.placeholder(dtype=tf.float32, shape=[], name='val_loss')
        val_summary_op = tf.summary.scalar('validation_loss', val_loss)

        return train_summary_op, summary_writer, saver, val_loss, val_summary_op

    def get_batch_generator(self, mode='train', resize=False):
        if mode == 'train':
            img_file = self.params['train_images_file']
        elif mode == 'validate':
            img_file = self.params['validation_images_file']
        else:
            raise NotImplementedError

        with open(self.params['data_path'] + img_file) as f:
            image_files = [k.rstrip() for k in f.readlines()]

        begin = 0
        while True:
            end = begin + self.params['batch_size']
            if end < len(image_files):
                batch_files = image_files[begin:end]
            else:
                end = end - len(image_files)
                batch_files = image_files[begin:] + image_files[:end]
            begin = end
            if resize:
                batch_paths = [self.params['data_path'] + 'images/' + k for k in batch_files]
                images = []
                for img_path in batch_paths:
                    image = filehandling_utils.load_image(img_path, resize=True)
                    if len(image.shape) == 2:
                        image = grey2rgb(image)
                    images.append(image)
                mat = np.stack(images, axis=0)
            else:
                batch_paths = [self.params['data_path'] + 'images_resized/' +
                               k[:-len('JPEG')] + 'bmp' for k in batch_files]
                images = []
                for img_path in batch_paths:
                    image = filehandling_utils.load_image(img_path, resize=False)
                    images.append(image)
                mat = np.stack(images, axis=0)
            yield mat

    def train_pre_image(self, image_path, grad_clip=100.0, lr_lower_points=()):
        """
        like mahendran & vedaldi, optimizes pre-image based on a single other image
        """

        if not os.path.exists(self.params['log_path']):
            os.makedirs(self.params['log_path'])

        save_dict(self.params, self.params['log_path'] + 'params.txt')

        with tf.Graph().as_default() as graph:
            with tf.Session() as sess:
                img_mat = filehandling_utils.load_image(image_path, resize=False)
                image = tf.constant(img_mat, dtype=tf.float32, shape=[1, self.img_hw, self.img_hw, 3])
                rec_init = tf.abs(tf.random_normal([1, self.img_hw, self.img_hw, 3], mean=0, stddev=0.0001))
                reconstruction = tf.get_variable('reconstruction', dtype=tf.float32,
                                                 initializer=rec_init)

                jitter_x_pl = tf.placeholder(dtype=tf.int32, shape=[], name='jitter_x_pl')
                jitter_y_pl = tf.placeholder(dtype=tf.int32, shape=[], name='jitter_y_pl')

                rec_part = tf.slice(reconstruction, [0, jitter_x_pl, jitter_y_pl, 0], [-1, -1, -1, -1])
                rec_padded = tf.pad(rec_part, paddings=[[0, 0], [jitter_x_pl, 0], [jitter_y_pl, 0], [0, 0]])

                net_input = tf.concat([image, rec_padded], axis=0)

                self.load_classifier(net_input)

                # representation = graph.get_tensor_by_name(params['layer_name'])

                # if len(representation.get_shape()) == 2:
                #     representation = tf.reshape(representation, shape=[2, -1, 1, 1])

                # img_rep = tf.slice(representation, [0, 0, 0, 0], [1, -1, -1, -1])
                # rec_rep = tf.slice(representation, [1, 0, 0, 0], [1, -1, -1, -1])

                # mse_mod = NormedMSELoss(target=img_rep.name, reconstruction=rec_rep.name, weighting=params['mse_C'])
                # sr_mod = SoftRangeLoss(reconstruction.name, params['alpha_sr'],
                #                        weighting=1 / (params['img_HW'] ** 2 * params['range_B'] ** params['alpha_sr']))
                # tv_mod = TotalVariationLoss(reconstruction.name, params['beta_tv'],
                #                             weighting=1 / (
                #                             params['img_HW'] ** 2 * params['range_V'] ** params['beta_tv']))
                #
                # ica_prior = ICAPrior(tensor_names='reconstruction/read:0',
                #                      weighting=1.0e-5, name='ICAPrior',
                #                      load_path='./logs/priors/ica_prior/8by8_512_color/ckpt-10000',
                #                      trainable=False, filter_dims=[8, 8], input_scaling=1.0, n_components=512)

                loss, train_op = self.build_model()
                lr_pl = tf.placeholder(dtype=tf.float32, shape=[])
                # optimizer = tf.train.AdamOptimizer(lr_pl)
                optimizer = tf.train.MomentumOptimizer(lr_pl, momentum=0.9)
                tvars = tf.trainable_variables()
                grads = tf.gradients(loss, tvars)
                tg_pairs = [k for k in zip(grads, tvars) if k[0] is not None]
                tg_clipped = [(tf.clip_by_value(k[0], -grad_clip, grad_clip), k[1])
                              for k in tg_pairs]
                train_op = optimizer.apply_gradients(tg_clipped)

                clip_op = tf.assign(reconstruction, tf.clip_by_value(reconstruction, 0.0, 2.0 * params['range_B']))

                train_summary_op, summary_writer, saver, val_loss, val_summary_op = self.build_logging(loss)

                sess.run(tf.global_variables_initializer())

                for mod in self.params['loss_modules']:
                    if isinstance(mod, LearnedPriorLoss):
                        mod.load_weights(sess)

                start_time = time.time()
                for count in range(self.params['num_iterations']):
                    jitter = np.random.randint(low=0, high=params['jitter_T'] + 1, dtype=int, size=(2,))

                    if (count + 1) % params['lr_lower_freq'] == 0:
                        self.params['learning_rate'] = self.params['learning_rate'] / 10.0
                        print('lowering learning rate to {0}'.format(self.params['learning_rate']))

                    feed = {lr_pl: self.params['learning_rate'], jitter_x_pl: jitter[0], jitter_y_pl: jitter[1]}

                    batch_loss, _, summary_string = sess.run([loss, train_op, train_summary_op], feed_dict=feed)
                    # sess.run(check, feed_dict=feed)
                    # sess.run(clip_op)
                    rec_mat = sess.run(reconstruction)
                    if (count + 1) % params['summary_freq'] == 0:
                        summary_writer.add_summary(summary_string, count)

                    if (count + 1) % params['print_freq'] == 0:
                        print(('Iteration: {0:6d} Training Error:   {1:9.2f} ' +
                               'Time: {2:5.1f} min').format(count + 1, batch_loss, (time.time() - start_time) / 60))
                        # a = sess.run(graph.get_tensor_by_name('ICAPrior/ica_a:0'))
                        # print(a)
                        # w = sess.run(graph.get_tensor_by_name('ICAPrior/ica_w:0'))
                        # print(w)

                    if (count + 1) % params['log_freq'] == 0:
                        plot_mat = np.zeros(shape=(self.img_hw, 2 * self.img_hw, 3))

                        plot_mat[:, :self.img_hw, :] = img_mat / 255.0
                        rec_mat = (rec_mat - np.min(rec_mat)) / (np.max(rec_mat) - np.min(rec_mat))  # M&V just rescale
                        plot_mat[:, self.img_hw:, :] = rec_mat
                        if params['save_as_mat']:
                            np.save(self.params['log_path'] + 'rec_' + str(count + 1) + '.npy', plot_mat)
                        else:
                            fig = plt.figure(frameon=False)
                            fig.set_size_inches(2, 1)
                            ax = plt.Axes(fig, [0., 0., 1., 1.])
                            ax.set_axis_off()
                            fig.add_axes(ax)
                            ax.imshow(plot_mat, aspect='auto')
                            plt.savefig(self.params['log_path'] + 'rec_' + str(count + 1) + '.png',
                                        format='png', dpi=self.img_hw)

    def train_on_dataset(self):
        """
        trains all trainable variables with respect to the registered losses on the imagenet validation set
        """

        if not os.path.exists(self.params['log_path']):
            os.makedirs(self.params['log_path'])

        save_dict(self.params, self.params['log_path'] + 'params.txt')

        batch_gen = self.get_batch_generator(mode='train')
        with tf.Graph().as_default():
            with tf.Session() as sess:
                img_pl = tf.placeholder(dtype=tf.float32, shape=[self.params['batch_size'], self.img_hw,
                                                                 self.img_hw, self.img_channels])

                self.load_classifier(img_pl)
                loss, train_op = self.build_model()
                train_summary_op, summary_writer, saver, val_loss, val_summary_op = self.build_logging(loss)

                for inv_mod in self.params['inv_modules']:
                    if isinstance(inv_mod, LearnedPriorLoss) and inv_mod.trainable is False:
                        inv_mod.load_weights(sess)

                for loss_mod in self.params['loss_modules']:
                    if isinstance(loss_mod, LearnedPriorLoss) and loss_mod.trainable is False:
                        loss_mod.load_weights(sess)

                if self.params['load_path']:
                    global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                    opt_vars = [v for v in global_vars if v not in train_vars]

                    if self.params['load_opt_vars']:
                        tf.train.Saver(var_list=opt_vars).restore(sess, self.params['load_path'])
                    else:
                        sess.run(tf.variables_initializer(opt_vars))

                    load_vars = tf.trainable_variables()
                    loader = tf.train.Saver(var_list=load_vars)
                    loader.restore(sess, self.params['load_path'])

                else:
                    sess.run(tf.global_variables_initializer())

                start_time = time.time()
                train_time = 0.0
                for count in range(self.params['num_iterations']):
                    feed_dict = {img_pl: next(batch_gen)}

                    batch_start = time.time()
                    batch_loss, _, summary_string = sess.run([loss, train_op, train_summary_op],
                                                             feed_dict=feed_dict)
                    train_time += time.time() - batch_start

                    if (count + 1) % self.params['summary_freq'] == 0:
                        summary_writer.add_summary(summary_string, count)

                    if (count + 1) % self.params['print_freq'] == 0:
                        summary_writer.flush()
                        print(('Iteration: {0:6d} Training Error:   {1:9.2f} ' +
                               'Time: {2:5.1f} min').format(count + 1, batch_loss, (time.time() - start_time) / 60))

                    if (count + 1) % self.params['log_freq'] == 0 or (count + 1) == self.params['num_iterations']:
                        checkpoint_file = os.path.join(self.params['log_path'], 'ckpt')
                        saver.save(sess, checkpoint_file, global_step=(count + 1), write_meta_graph=False)

                    if self.params['test_freq'] > 0 and ((count + 1) % self.params['test_freq'] == 0 or
                                                         (count + 1) == self.params['num_iterations']):
                        val_batch_gen = self.get_batch_generator(mode='validate')
                        val_loss_acc = 0.0
                        num_runs = self.params['test_set_size'] // self.params['batch_size'] + 1
                        for val_count in range(num_runs):
                            val_feed_dict = {img_pl: next(val_batch_gen)}
                            val_batch_loss = sess.run(loss, feed_dict=val_feed_dict)
                            val_loss_acc += val_batch_loss
                        val_loss_acc /= num_runs
                        val_summary_string = sess.run(val_summary_op, feed_dict={val_loss: val_loss_acc})
                        summary_writer.add_summary(val_summary_string, count)
                        print(('Iteration: {0:6d} Validation Error: {1:9.2f} ' +
                               'Time: {2:5.1f} min').format(count + 1, val_loss_acc, (time.time() - start_time) / 60))
                sess_time = time.time() - start_time
                train_ratio = 100.0 * train_time / sess_time
                print('Session finished. {0:2.1f}% of the time spent in run calls'.format(train_ratio))

    def run_inverse_model(self, inv_input_mat, ckpt_num=3000):
        """
        runs the model on a specified input. used for stacking models
        :param inv_input_mat: numpy array serving as input
        :param ckpt_num: checkpoint number to be loaded
        :return: numpy array produced by the model
        """
        with tf.Graph().as_default() as graph:
            with tf.Session() as sess:
                img_pl = tf.placeholder(dtype=tf.float32,
                                        shape=[self.params['batch_size'], self.img_hw, self.img_hw, self.img_channels])
                self.load_classifier(img_pl)

                if not isinstance(self.params['inv_modules'], list):
                    self.params['inv_modules'] = [self.params['inv_modules']]
                input_shape = graph.get_tensor_by_name(self.params['inv_modules'][0]['inv_input_name']).get_shape()
                inv_input = tf.placeholder(dtype=tf.float32, name='new_input', shape=input_shape)
                self.params['inv_modules'][0]['inv_input_name'] = 'new_input:0'

                self.build_model()

                saver = tf.train.Saver()
                saver.restore(sess, self.params['log_path'] + 'ckpt-' + str(ckpt_num))
                rec_tensor_name = 'module_' + str(len(self.params['inv_modules'])) + '/reconstruction:0'
                reconstruction = graph.get_tensor_by_name(rec_tensor_name)
                rec_mat = sess.run(reconstruction, feed_dict={inv_input: inv_input_mat})
                return rec_mat

    def visualize(self, num_images=7, rec_type='bgr_normed', file_name='img_vs_rec', ckpt_num=1000, add_diffs=True):
        """
        makes image file showing reconstrutions from a trained model
        :param num_images: first n images from the data set are visualized
        :param rec_type: details, which parts of network preprocessing need to be inverted
        :param file_name: name of output file
        :param ckpt_num: checkpoint to be loaded
        :param add_diffs: if true, 2 extra visualizations are added
        :return: None
        """

        actual_batch_size = self.params['batch_size']
        assert num_images <= actual_batch_size
        self.params['batch_size'] = num_images

        batch_gen = self.get_batch_generator(mode='validate')

        with tf.Graph().as_default() as graph:
            with tf.Session() as sess:
                img_pl = tf.placeholder(dtype=tf.float32,
                                        shape=[self.params['batch_size'], self.img_hw, self.img_hw, self.img_channels])
                self.load_classifier(img_pl)
                self.build_model()
                saver = tf.train.Saver()

                saver.restore(sess, self.params['log_path'] + 'ckpt-' + str(ckpt_num))
                feed_dict = {img_pl: next(batch_gen)}
                rec_tensor_name = 'module_' + str(len(self.params['inv_modules'])) + '/reconstruction:0'
                reconstruction = graph.get_tensor_by_name(rec_tensor_name)
                rec_mat = sess.run(reconstruction, feed_dict=feed_dict)

        self.params['batch_size'] = actual_batch_size

        img_mat = feed_dict[img_pl] / 255.0

        if rec_type == 'rgb_scaled':
            rec_mat /= 255.0
        elif rec_type == 'bgr_normed':
            rec_mat = rec_mat[:, :, :, ::-1]
            if self.params['classifier'].lower() == 'vgg16':
                rec_mat = rec_mat + self.imagenet_mean
            elif self.params['classifier'].lower() == 'alexnet':
                rec_mat = rec_mat + np.mean(self.imagenet_mean)
            else:
                raise NotImplementedError
            rec_mat /= 255.0
        else:
            raise NotImplementedError

        print('reconstruction min and max vals: ' + str(rec_mat.min()) + ', ' + str(rec_mat.max()))
        rec_mat = np.minimum(np.maximum(rec_mat, 0.0), 1.0)

        if add_diffs:
            cols = 4
        else:
            cols = 2

        plot_mat = np.zeros(shape=(rec_mat.shape[0]*rec_mat.shape[1], rec_mat.shape[2]*cols, 3))
        for idx in range(rec_mat.shape[0]):
            h = rec_mat.shape[1]
            w = rec_mat.shape[2]
            plot_mat[idx * h:(idx + 1) * h, :w, :] = img_mat[idx, :, :, :]
            plot_mat[idx * h:(idx + 1) * h, w:2 * w, :] = rec_mat[idx, :, :, :]
            if add_diffs:
                diff = img_mat[idx, :, :, :] - rec_mat[idx, :, :, :]
                diff -= np.min(diff)
                diff /= np.max(diff)
                plot_mat[idx * h:(idx + 1) * h, 2 * w:3 * w, :] = diff
                abs_diff = np.abs(rec_mat[idx, :, :, :] - img_mat[idx, :, :, :])
                abs_diff /= np.max(abs_diff)
                plot_mat[idx * h:(idx + 1) * h, 3 * w:, :] = abs_diff

        fig = plt.figure(frameon=False)
        fig.set_size_inches(cols, num_images)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(plot_mat, aspect='auto')
        plt.savefig(self.params['log_path'] + file_name + '.png', format='png', dpi=224)
