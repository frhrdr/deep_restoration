import tensorflow as tf
from tf_vgg import vgg16
from tf_alexnet import alexnet
from utils.filehandling import load_image
from utils.temp_utils import get_optimizer, plot_feat_map_diffs
from modules.core_modules import LossModule, LearnedPriorLoss, TrainedModule
import os
import time
import numpy as np
from skimage.color import grey2rgb
import matplotlib
matplotlib.use('tkagg', force=True)
import matplotlib.pyplot as plt


class NetInversion:

    def __init__(self, modules, log_path, classifier='alexnet', summary_freq=10, print_freq=50, log_freq=500):
        assert classifier in ('alexnet', 'vgg16')
        self.modules = modules
        self.log_path = log_path
        self.classifier = classifier
        self.print_freq = print_freq
        self.log_freq = log_freq
        self.summary_freq = summary_freq
        self.imagenet_mean = np.asarray([123.68, 116.779, 103.939])  # in RGB order
        if classifier == 'alexnet':
            self.img_hw = 227
        else:
            self.img_hw = 224
        self.img_channels = 3

    def load_classifier(self, img_pl):
        if self.classifier.lower() == 'vgg16':
            classifier = vgg16.Vgg16()
        elif self.classifier.lower() == 'alexnet':
            classifier = alexnet.AlexNet()
        else:
            raise NotImplementedError

        classifier.build(img_pl, rescale=1.0)

    def load_partial_classifier(self, in_tensor, in_tensor_name):
        if self.classifier.lower() == 'vgg16':
            raise NotImplementedError
            # classifier = vgg16.Vgg16()
        elif self.classifier.lower() == 'alexnet':
            classifier = alexnet.AlexNet()
        else:
            raise NotImplementedError

        classifier.build_partial(in_tensor, in_tensor_name, rescale=1.0)

    def build_model(self):
        loss = 0
        for idx, mod in enumerate(self.modules):
            mod.build(scope_suffix=str(idx))
            if isinstance(mod, LossModule):
                loss += mod.get_loss()

        return loss

    def build_logging(self, loss):
        tf.summary.scalar('Total_Loss', loss)
        for mod in self.modules:
            if isinstance(mod, LossModule):
                mod.scalar_summary()

        train_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.log_path + '/summaries')
        saver = tf.train.Saver()

        val_loss = tf.placeholder(dtype=tf.float32, shape=[], name='val_loss')
        val_summary_op = tf.summary.scalar('validation_loss', val_loss)

        return train_summary_op, summary_writer, saver, val_loss, val_summary_op

    def get_batch_generator(self, batch_size, mode, resize=False,
                            data_path='../data/imagenet2012-validationset/',
                            train_images_file='train_48k_images.txt',
                            validation_images_file='validate_2k_images.txt'):
        if mode == 'train':
            img_file = train_images_file
        elif mode == 'validate':
            img_file = validation_images_file
        else:
            raise NotImplementedError

        if self.classifier.lower() == 'alexnet':
            subdir = 'images_resized_227/'
        elif self.classifier.lower() == 'vgg16':
            subdir = 'images_resized_224/'
        else:
            raise NotImplementedError

        with open(data_path + img_file) as f:
            image_files = [k.rstrip() for k in f.readlines()]

        begin = 0
        while True:
            end = begin + batch_size
            if end < len(image_files):
                batch_files = image_files[begin:end]
            else:
                end = end - len(image_files)
                batch_files = image_files[begin:] + image_files[:end]
            begin = end
            if resize:
                batch_paths = [data_path + 'images/' + k for k in batch_files]
                images = []
                for img_path in batch_paths:
                    image = load_image(img_path, resize=True)
                    if len(image.shape) == 2:
                        image = grey2rgb(image)
                    images.append(image)
                mat = np.stack(images, axis=0)
            else:
                batch_paths = [data_path + subdir + k[:-len('JPEG')] + 'bmp' for k in batch_files]
                images = []
                for img_path in batch_paths:
                    image = load_image(img_path, resize=False)
                    images.append(image)
                mat = np.stack(images, axis=0)
            yield mat

    def train_pre_featmap(self, image_path, n_iterations, grad_clip=100.0, lr_lower_points=((0, 1e-4),),
                          range_b=80, jitter_t=0, optim_name='adam',
                          range_clip=False, save_as_plot=False, jitter_stop_point=-1, scale_pre_img=1.0,
                          pre_featmap_init=None, ckpt_offset=0,
                          pre_featmap_name='input',
                          tensor_names_to_save=(), featmap_names_to_plot=(), max_n_featmaps_to_plot=5):
        """
        like mahendran & vedaldi, optimizes pre-image based on a single other image
        """

        if not os.path.exists(self.log_path + 'mats/'):
            os.makedirs(self.log_path + 'mats/')
        if save_as_plot and not os.path.exists(self.log_path + 'imgs/'):
            os.makedirs(self.log_path + 'imgs/')

        img_mat = np.expand_dims(load_image(image_path, resize=False), axis=0)

        target_featmap_mat = self.get_target_featmap(img_mat, pre_featmap_name)
        print('target_shape:', target_featmap_mat.shape)

        with tf.Graph().as_default() as graph:
            with tf.Session() as sess:
                target_featmap = tf.get_variable(name='target_featmap', dtype=tf.float32, trainable=False,
                                                 initializer=target_featmap_mat)
                print(target_featmap)

                if pre_featmap_init is None and scale_pre_img == 2.7098e+4:
                    print('using special initializer')  # remnant from m&v settings
                    pre_featmap_init = tf.abs(tf.random_normal([1, self.img_hw, self.img_hw, 3], mean=0, stddev=0.0001))
                elif pre_featmap_init is None and pre_featmap_name == 'input':
                    pre_featmap_init = np.random.normal(loc=np.mean(self.imagenet_mean), scale=1.1,
                                                        size=([1, self.img_hw, self.img_hw, 3])).astype(np.float32)
                    pre_featmap_init = np.maximum(pre_featmap_init, 100.)
                    pre_featmap_init = np.minimum(pre_featmap_init, 155.)
                elif pre_featmap_init is None:
                    pre_featmap_init = np.random.normal(loc=0, scale=0.1,
                                                        size=target_featmap_mat.shape).astype(np.float32)
                    # think about taking mean and sdev from target feature map layers
                pre_featmap = tf.get_variable('pre_featmap', dtype=tf.float32, initializer=pre_featmap_init)

                jitter_x_pl = tf.placeholder(dtype=tf.int32, shape=[], name='jitter_x_pl')
                jitter_y_pl = tf.placeholder(dtype=tf.int32, shape=[], name='jitter_y_pl')

                rec_part = tf.slice(pre_featmap, [0, jitter_x_pl, jitter_y_pl, 0], [-1, -1, -1, -1])
                rec_padded = tf.pad(rec_part, paddings=[[0, 0], [jitter_x_pl, 0], [jitter_y_pl, 0], [0, 0]])

                use_jitter_pl = tf.placeholder(dtype=tf.bool, shape=[], name='use_jitter')
                rec_input = tf.cond(use_jitter_pl, lambda: rec_padded, lambda: pre_featmap, name='jitter_cond')
                net_input = tf.concat([target_featmap, rec_input * scale_pre_img], axis=0)

                self.load_partial_classifier(net_input, pre_featmap_name)

                loss = self.build_model()
                tensors_to_save = [graph.get_tensor_by_name(k) for k in tensor_names_to_save]
                featmaps_to_plot = [graph.get_tensor_by_name(k) for k in featmap_names_to_plot]

                if optim_name.lower() == 'l-bfgs-b':
                    options = {'maxiter': n_iterations}
                    scipy_opt = tf.contrib.opt.ScipyOptimizerInterface(loss, method='L-BFGS-B',
                                                                       options=options)

                    def loss_cb(*args):
                        print(args)

                    sess.run(tf.global_variables_initializer())
                    scipy_opt.minimize(session=sess, feed_dict={jitter_x_pl: 0, jitter_y_pl: 0, use_jitter_pl: False},
                                       loss_callback=loss_cb)
                    rec_mat = sess.run(pre_featmap)
                    np.save(self.log_path + 'mats/rec_{}.npy'.format(n_iterations), rec_mat)
                else:
                    lr_pl = tf.placeholder(dtype=tf.float32, shape=[])
                    optimizer = get_optimizer(optim_name, lr_pl)

                    tvars = tf.trainable_variables()
                    grads = tf.gradients(loss, tvars)
                    tg_pairs = [k for k in zip(grads, tvars) if k[0] is not None]
                    tg_clipped = [(tf.clip_by_value(k[0], -grad_clip, grad_clip), k[1])
                                  for k in tg_pairs]
                    train_op = optimizer.apply_gradients(tg_clipped)

                    if range_clip and pre_featmap_name == 'input':
                        position_norm = tf.sqrt(tf.reduce_sum((pre_featmap - self.imagenet_mean) ** 2, axis=3))
                        box_rescale = tf.minimum(2 * range_b / position_norm, 1.)
                        box_rescale = tf.stack([box_rescale] * 3, axis=3)
                        clip_op = tf.assign(pre_featmap,
                                            (pre_featmap - self.imagenet_mean) * box_rescale + self.imagenet_mean)
                    else:
                        clip_op = None

                    train_summary_op, summary_writer, saver, val_loss, val_summary_op = self.build_logging(loss)

                    sess.run(tf.global_variables_initializer())

                    for mod in self.modules:
                        if isinstance(mod, (TrainedModule, LearnedPriorLoss)):
                            mod.load_weights(sess)

                    use_jitter = False if jitter_t == 0 else True
                    lr = lr_lower_points[0][1]
                    start_time = time.time()
                    for count in range(ckpt_offset + 1, ckpt_offset + n_iterations + 1):
                        jitter = np.random.randint(low=0, high=jitter_t + 1, dtype=int, size=(2,))

                        feed = {lr_pl: lr,
                                jitter_x_pl: jitter[0],
                                jitter_y_pl: jitter[1],
                                use_jitter_pl: use_jitter}

                        batch_loss, _, summary_string = sess.run([loss, train_op, train_summary_op], feed_dict=feed)

                        if range_clip:
                            sess.run(clip_op)

                        if count % self.summary_freq == 0:
                            summary_writer.add_summary(summary_string, count)

                        if count % self.print_freq == 0:
                            print(('Iteration: {0:6d} Training Error:   {1:8.5f} ' +
                                   'Time: {2:5.1f} min').format(count, batch_loss, (time.time() - start_time) / 60))

                        if count % self.log_freq == 0:
                            rec_mat = sess.run(pre_featmap)
                            np.save(self.log_path + 'mats/rec_' + str(count) + '.npy', rec_mat)

                            if save_as_plot:
                                plot_mat = np.zeros(shape=(self.img_hw, 2 * self.img_hw, 3))
                                plot_mat[:, :self.img_hw, :] = img_mat / 255.0
                                rec_mat = (rec_mat - np.min(rec_mat)) / (np.max(rec_mat) - np.min(rec_mat))
                                plot_mat[:, self.img_hw:, :] = rec_mat

                                fig = plt.figure(frameon=False)
                                fig.set_size_inches(2, 1)
                                ax = plt.Axes(fig, [0., 0., 1., 1.])
                                ax.set_axis_off()
                                fig.add_axes(ax)
                                ax.imshow(plot_mat, aspect='auto')
                                plt.savefig(self.log_path + 'imgs/rec_' + str(count) + '.png',
                                            format='png', dpi=self.img_hw)
                                plt.close()

                            if tensors_to_save:
                                mats_to_save = sess.run(tensors_to_save, feed_dict=feed)

                                for idx, fmap in enumerate(mats_to_save):
                                    file_name = tensor_names_to_save[idx] + '-' + str(count) + '.npy'
                                    np.save(self.log_path + 'mats/' + file_name, fmap)

                            if featmaps_to_plot:
                                fmaps_to_plot = sess.run(featmaps_to_plot, feed_dict=feed)

                                for fmap, name in zip(fmaps_to_plot, featmap_names_to_plot):
                                    name = name.replace('/', '_').rstrip(':0')
                                    file_path = '{}mats/{}-{}.npy'.format(self.log_path, name, count)
                                    np.save(file_path, fmap)
                                    if save_as_plot:
                                        file_path = '{}imgs/{}-{}.png'.format(self.log_path, name, count)
                                        plot_feat_map_diffs(fmap, file_path, max_n_featmaps_to_plot)

                        if jitter_stop_point == count:
                            print('Jittering stopped at ', count)
                            use_jitter = False

                        if lr_lower_points and lr_lower_points[0][0] <= count:
                            lr = lr_lower_points[0][1]
                            print('new learning rate: ', lr)
                            lr_lower_points = lr_lower_points[1:]

    def train_on_dataset(self, n_iterations, batch_size, test_set_size=200, test_freq=100,
                         optim_name='adam', lr_lower_points=((0, 1e-4),)):
        """
        trains all trainable variables with respect to the registered losses on the imagenet validation set
        """

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        batch_gen = self.get_batch_generator(batch_size, mode='train')

        with tf.Graph().as_default():

            img_pl = tf.placeholder(dtype=tf.float32, shape=[batch_size, self.img_hw,
                                                             self.img_hw, self.img_channels])

            self.load_classifier(img_pl)
            loss = self.build_model()

            lr_pl = tf.placeholder(dtype=tf.float32, shape=[])
            optimizer = get_optimizer(optim_name, lr_pl)
            train_op = optimizer.minimize(loss)

            train_summary_op, summary_writer, saver, val_loss, val_summary_op = self.build_logging(loss)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                for inv_mod in self.modules:
                    if (isinstance(inv_mod, LearnedPriorLoss) or isinstance(inv_mod, TrainedModule)) \
                       and inv_mod.trainable is False:
                        inv_mod.load_weights(sess)

                lr = lr_lower_points[0][1]
                train_time = 0.0
                start_time = time.time()
                for count in range(1, n_iterations + 1):
                    feed_dict = {img_pl: next(batch_gen), lr_pl: lr}

                    batch_start = time.time()
                    batch_loss, _, summary_string = sess.run([loss, train_op, train_summary_op],
                                                             feed_dict=feed_dict)
                    train_time += time.time() - batch_start

                    if count % self.summary_freq == 0:
                        summary_writer.add_summary(summary_string, count)

                    if count % self.print_freq == 0:
                        summary_writer.flush()
                        print(('Iteration: {0:6d} Training Error:   {1:9.2f} ' +
                               'Time: {2:5.1f} min').format(count, batch_loss, (time.time() - start_time) / 60))

                    if count % self.log_freq == 0 or count == n_iterations:
                        # checkpoint_file = os.path.join(self.log_path, 'ckpt')
                        # saver.save(sess, checkpoint_file, global_step=(count + 1), write_meta_graph=False)
                        for inv_mod in self.modules:
                            if isinstance(inv_mod, TrainedModule) and inv_mod.trainable is True:
                                inv_mod.save_weights(sess, count)

                    if test_freq > 0 and (count % test_freq == 0 or count == n_iterations):
                        val_batch_gen = self.get_batch_generator(batch_size, mode='validate')
                        val_loss_acc = 0.0
                        num_runs = test_set_size // batch_size + 1
                        for val_count in range(num_runs):
                            val_feed_dict = {img_pl: next(val_batch_gen)}
                            val_batch_loss = sess.run(loss, feed_dict=val_feed_dict)
                            val_loss_acc += val_batch_loss
                        val_loss_acc /= num_runs
                        val_summary_string = sess.run(val_summary_op, feed_dict={val_loss: val_loss_acc})
                        summary_writer.add_summary(val_summary_string, count)
                        print(('Iteration: {0:6d} Validation Error: {1:9.2f} ' +
                               'Time: {2:5.1f} min').format(count, val_loss_acc, (time.time() - start_time) / 60))

                    if lr_lower_points and lr_lower_points[0][0] <= count:
                        lr = lr_lower_points[0][1]
                        print('new learning rate: ', lr)
                        lr_lower_points = lr_lower_points[1:]

                sess_time = time.time() - start_time
                train_ratio = 100.0 * train_time / sess_time
                print('Session finished. {0:2.1f}% of the time spent in run calls'.format(train_ratio))

    def get_target_featmap(self, target_image_mat, target_map_name):
        if target_map_name == 'input':
            return target_image_mat
        else:
            with tf.Graph().as_default() as graph:
                image = tf.constant(target_image_mat, dtype=tf.float32, shape=[1, self.img_hw, self.img_hw, 3])
                self.load_classifier(image)
                feat_map = graph.get_tensor_by_name(target_map_name + ':0')
                with tf.Session() as sess:
                    feat_map_mat = sess.run(feat_map)

            return feat_map_mat

    def run_model_on_image(self, image_file, tensor_names_to_fetch):
        img_mat = np.expand_dims(load_image(image_file, resize=False), axis=0)

        with tf.Graph().as_default() as graph:

            img_tensor = tf.constant(img_mat, dtype=tf.float32)

            self.load_classifier(img_tensor)
            self.build_model()
            tensors_to_fetch = [graph.get_tensor_by_name(n) for n in tensor_names_to_fetch]

            with tf.Session() as sess:
                for inv_mod in self.modules:
                    if (isinstance(inv_mod, LearnedPriorLoss) or isinstance(inv_mod, TrainedModule)) \
                       and inv_mod.trainable is False:
                        inv_mod.load_weights(sess)

                fetched_mats = sess.run(tensors_to_fetch)

        return fetched_mats

    # def visualize(self, num_images=7, rec_type='bgr_normed', file_name='img_vs_rec', ckpt_num=1000, add_diffs=True):
    #     """
    #     makes image file showing reconstrutions from a trained model
    #     :param num_images: first n images from the data set are visualized
    #     :param rec_type: details, which parts of network preprocessing need to be inverted
    #     :param file_name: name of output file
    #     :param ckpt_num: checkpoint to be loaded
    #     :param add_diffs: if true, 2 extra visualizations are added
    #     :return: None
    #     """
    #
    #     actual_batch_size = None  # batch_size
    #     assert num_images <= actual_batch_size
    #     batch_size = num_images
    #
    #     batch_gen = self.get_batch_generator(batch_size, mode='validate')
    #
    #     with tf.Graph().as_default() as graph:
    #         with tf.Session() as sess:
    #             img_pl = tf.placeholder(dtype=tf.float32,
    #                                     shape=[batch_size, self.img_hw, self.img_hw, self.img_channels])
    #             self.load_classifier(img_pl)
    #             self.build_model()
    #             saver = tf.train.Saver()
    #
    #             saver.restore(sess, self.log_path + 'ckpt-' + str(ckpt_num))
    #             feed_dict = {img_pl: next(batch_gen)}
    #             rec_tensor_name = 'module_' + str(len(self.inv_modules)) + '/reconstruction:0'
    #             reconstruction = graph.get_tensor_by_name(rec_tensor_name)
    #             rec_mat = sess.run(reconstruction, feed_dict=feed_dict)
    #
    #     batch_size = actual_batch_size
    #
    #     img_mat = feed_dict[img_pl] / 255.0
    #
    #     if rec_type == 'rgb_scaled':
    #         rec_mat /= 255.0
    #     elif rec_type == 'bgr_normed':
    #         rec_mat = rec_mat[:, :, :, ::-1]
    #         if self.classifier.lower() == 'vgg16':
    #             rec_mat = rec_mat + self.imagenet_mean
    #         elif self.classifier.lower() == 'alexnet':
    #             rec_mat = rec_mat + np.mean(self.imagenet_mean)
    #         else:
    #             raise NotImplementedError
    #         rec_mat /= 255.0
    #     else:
    #         raise NotImplementedError
    #
    #     print('reconstruction min and max vals: ' + str(rec_mat.min()) + ', ' + str(rec_mat.max()))
    #     rec_mat = np.minimum(np.maximum(rec_mat, 0.0), 1.0)
    #
    #     if add_diffs:
    #         cols = 4
    #     else:
    #         cols = 2
    #
    #     plot_mat = np.zeros(shape=(rec_mat.shape[0]*rec_mat.shape[1], rec_mat.shape[2]*cols, 3))
    #     for idx in range(rec_mat.shape[0]):
    #         h = rec_mat.shape[1]
    #         w = rec_mat.shape[2]
    #         plot_mat[idx * h:(idx + 1) * h, :w, :] = img_mat[idx, :, :, :]
    #         plot_mat[idx * h:(idx + 1) * h, w:2 * w, :] = rec_mat[idx, :, :, :]
    #         if add_diffs:
    #             diff = img_mat[idx, :, :, :] - rec_mat[idx, :, :, :]
    #             diff -= np.min(diff)
    #             diff /= np.max(diff)
    #             plot_mat[idx * h:(idx + 1) * h, 2 * w:3 * w, :] = diff
    #             abs_diff = np.abs(rec_mat[idx, :, :, :] - img_mat[idx, :, :, :])
    #             abs_diff /= np.max(abs_diff)
    #             plot_mat[idx * h:(idx + 1) * h, 3 * w:, :] = abs_diff
    #
    #     fig = plt.figure(frameon=False)
    #     fig.set_size_inches(cols, num_images)
    #     ax = plt.Axes(fig, [0., 0., 1., 1.])
    #     ax.set_axis_off()
    #     fig.add_axes(ax)
    #     ax.imshow(plot_mat, aspect='auto')
    #     plt.savefig(self.log_path + file_name + '.png', format='png', dpi=224)
