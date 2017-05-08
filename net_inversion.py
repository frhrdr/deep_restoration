import tensorflow as tf
from tf_vgg import vgg16
from tf_alexnet import alexnet
import filehandling_utils
from parameter_utils import check_params
from filehandling_utils import save_dict
from model_utils import conv_deconv_model, deconv_conv_model, deconv_deconv_model
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

        classifier.build(img_pl)

    def build_model(self):
        graph = tf.get_default_graph()

        if not isinstance(self.params['inv_model_specs'], list):
            self.params['inv_model_specs'] = [self.params['inv_model_specs']]

        loss = 0.0

        for idx, spec in enumerate(self.params['inv_model_specs']):
            inv_input = graph.get_tensor_by_name(spec['inv_input_name'])
            inv_target = graph.get_tensor_by_name(spec['inv_target_name'])
            inv_target_shape = [k.value for k in inv_target.get_shape()[1:]]
            with tf.variable_scope('module_' + str(idx + 1)):
                if self.params['inv_model_type'].lower() == 'conv_deconv':
                    reconstruction = conv_deconv_model(inv_input, spec, inv_target_shape)
                elif self.params['inv_model_type'].lower() == 'deconv_conv':
                    reconstruction = deconv_conv_model(inv_input, spec, inv_target_shape)
                elif self.params['inv_model_type'].lower() == 'deconv_deconv':
                    reconstruction = deconv_deconv_model(inv_input, spec, inv_target_shape)
                else:
                    raise NotImplementedError

                reconstruction = tf.identity(reconstruction, name=spec['rec_name'])

                if spec['add_loss']:
                    loss += tf.losses.mean_squared_error(inv_target, reconstruction)

        if self.params['optimizer'].lower() == 'adam':
            train_op = tf.train.AdamOptimizer(learning_rate=self.params['learning_rate']).minimize(loss)
        else:
            raise NotImplementedError

        return loss, train_op

    def build_logging(self, graph, loss):
        tf.summary.scalar('mse_loss', loss)
        train_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.params['log_path'] + '/summaries', graph)
        saver = tf.train.Saver()

        val_loss = tf.placeholder(dtype=tf.float32, shape=[], name='val_loss')
        val_summary_op = tf.summary.scalar('validation_loss', val_loss)

        return train_summary_op, summary_writer, saver, val_loss, val_summary_op

    def get_batch_generator(self, mode='train'):
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

            batch_paths = [self.params['data_path'] + 'images/' + k for k in batch_files]
            images = []
            for img_path in batch_paths:
                image = filehandling_utils.load_image(img_path)
                if len(image.shape) == 2:
                    image = grey2rgb(image)
                images.append(image)
            mat = np.stack(images, axis=0)
            yield mat

    def train(self):
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
                train_summary_op, summary_writer, saver, val_loss, val_summary_op = self.build_logging(sess.graph, loss)

                if self.params['load_path']:
                    saver.restore(sess, self.params['load_path'])
                else:
                    sess.run(tf.global_variables_initializer())

                start_time = time.time()
                for count in range(self.params['num_iterations']):
                    feed_dict = {img_pl: next(batch_gen)}
                    batch_loss, _, summary_string = sess.run([loss, train_op, train_summary_op],
                                                             feed_dict=feed_dict)
                    summary_writer.add_summary(summary_string, count)

                    if (count + 1) % self.params['print_freq'] == 0:
                        summary_writer.flush()
                        print(('Iteration: {0:6d} Training Error:   {1:8.3f} ' +
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
                        print(('Iteration: {0:6d} Validation Error: {1:8.3f} ' +
                               'Time: {2:5.1f} min').format(count + 1, val_loss_acc, (time.time() - start_time) / 60))

    def run_inverse_model(self, inv_input_mat, ckpt_num=3000):
        with tf.Graph().as_default() as graph:
            with tf.Session() as sess:
                img_pl = tf.placeholder(dtype=tf.float32,
                                        shape=[self.params['batch_size'], self.img_hw, self.img_hw, self.img_channels])
                self.load_classifier(img_pl)

                if not isinstance(self.params['inv_model_specs'], list):
                    self.params['inv_model_specs'] = [self.params['inv_model_specs']]
                input_shape = graph.get_tensor_by_name(self.params['inv_model_specs'][0]['inv_input_name']).get_shape()
                inv_input = tf.placeholder(dtype=tf.float32, name='new_input', shape=input_shape)
                self.params['inv_model_specs'][0]['inv_input_name'] = 'new_input'

                self.build_model()

                saver = tf.train.Saver()
                saver.restore(sess, self.params['log_path'] + 'ckpt-' + str(ckpt_num))
                reconstruction = graph.get_tensor_by_name('reconstruction:0')
                rec_mat = sess.run(reconstruction, feed_dict={inv_input: inv_input_mat})
                return rec_mat

    def visualize(self, num_images=7, rec_type='bgr_normed', file_name='img_vs_rec', ckpt_num=3000, add_diffs=True):
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
                reconstruction = graph.get_tensor_by_name('reconstruction:0')
                rec_mat = sess.run(reconstruction, feed_dict=feed_dict)

        self.params['batch_size'] = actual_batch_size

        img_mat = feed_dict[img_pl]

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

    def visualize_old(self, img_idx=0, rec_type='rgb_scaled', ckpt_num=3000):
        actual_batch_size = self.params['batch_size']
        assert img_idx + 1 <= actual_batch_size
        self.params['batch_size'] = img_idx + 1

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
                reconstruction = graph.get_tensor_by_name('reconstruction:0')
                rec_mat = sess.run(reconstruction, feed_dict=feed_dict)

            self.params['batch_size'] = actual_batch_size

            img_mat = feed_dict[img_pl][img_idx, :, :, :]
            rec_mat = rec_mat[img_idx, :, :, :]
            if rec_type == 'rgb_scaled':
                rec_mat /= 255.0
            elif rec_type == 'bgr_normed':
                rec_mat = rec_mat[:, :, ::-1]
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

            w = h = 4
            fig = plt.figure(frameon=False)
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(img_mat, aspect='auto')
            plt.savefig(self.params['log_path'] + 'img' + str(img_idx) + '.png', dpi=224 / 4, format='png')
            fig = plt.figure(frameon=False)
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(rec_mat, aspect='auto')
            plt.savefig(self.params['log_path'] + 'rec' + str(img_idx) + '.png', dpi=224 / 4, format='png')


def run_stacked_models(params_list, num_images=7, file_name='stacked_inversion'):
    # extract final feature representation
    for p in params_list:
        p['batch_size'] = num_images

    li = NetInversion(params_list[0])

    if params_list[0]['classifier'].lower() == 'vgg16':
        model = vgg16.Vgg16()
    elif params_list[0]['classifier'].lower() == 'alexnet':
        model = alexnet.AlexNet()
    else:
        raise NotImplementedError

    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:
            img_pl = tf.placeholder(dtype=tf.float32,
                                    shape=[li.params['batch_size'], li.img_hw, li.img_hw, li.img_channels])
            model.build(img_pl)
            inv_input_tensor = graph.get_tensor_by_name(params_list[0]['inv_input_name'])
            inv_target_tensor = graph.get_tensor_by_name(params_list[-1]['inv_target_name'])
            batch_gen = li.get_batch_generator(mode='validate')
            img_target = next(batch_gen)
            inv_input, inv_target = sess.run([inv_input_tensor, inv_target_tensor], feed_dict={img_pl: img_target})

    # pass through models
    for params in params_list:
        li = NetInversion(params)
        inv_input = li.run_inverse_model(inv_input)
        print(inv_input.shape)

    # visualize final reconstruction
    img_mat = img_target
    rec_mat = inv_input
    rec_type = 'bgr_normed'
    add_diffs = True

    if rec_type == 'rgb_scaled':
        rec_mat /= 255.0
    elif rec_type == 'bgr_normed':
        rec_mat = rec_mat[:, :, :, ::-1]
        if li.params['classifier'].lower() == 'vgg16':
            rec_mat = rec_mat + li.imagenet_mean
        elif li.params['classifier'].lower() == 'alexnet':
            rec_mat = rec_mat + np.mean(li.imagenet_mean)
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

    plot_mat = np.zeros(shape=(rec_mat.shape[0] * rec_mat.shape[1], rec_mat.shape[2] * cols, 3))
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
    plt.savefig(file_name + '.png', format='png', dpi=224)
