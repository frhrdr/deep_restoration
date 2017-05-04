import matplotlib
matplotlib.use('tkagg', force=True)
import matplotlib.pyplot as plt
import tensorflow as tf
from tf_vgg import vgg16
from tf_alexnet import alexnet
import filehandling_utils
from parameter_utils import check_params
from filehandling_utils import save_dict
import os
import time
import numpy as np
from skimage.color import grey2rgb


class LayerInversion:

    def __init__(self, params):
        check_params(params)
        self.params = params
        self.imagenet_mean = np.asarray([123.68, 116.779, 103.939])  # in RGB order
        self.img_hw = 224
        self.img_channels = 3

    def build_model(self, img_pl):
        if self.params['classifier'].lower() == 'vgg16':
            model = vgg16.Vgg16()
        elif self.params['classifier'].lower() == 'alexnet':
            model = alexnet.AlexNet()
        else:
            raise NotImplementedError

        model.build(img_pl)
        graph = tf.get_default_graph()

        self.inv_input = graph.get_tensor_by_name(self.params['inv_input_name'])
        self.inv_target = graph.get_tensor_by_name(self.params['inv_target_name'])
        self.inv_input_height = self.inv_input.get_shape()[1].value
        self.inv_input_width = self.inv_input.get_shape()[2].value
        self.inv_input_channels = self.inv_input.get_shape()[3].value
        self.inv_target_height = self.inv_target.get_shape()[1].value
        self.inv_target_width = self.inv_target.get_shape()[2].value
        self.inv_target_channels = self.inv_target.get_shape()[3].value

        if self.params['inv_model'].lower() == 'conv_deconv':
            self.reconstruction = self.conv_deconv_model(self.inv_input)
        elif self.params['inv_model'].lower() == 'deconv_conv':
            self.reconstruction = self.deconv_conv_model(self.inv_input)
        elif self.params['inv_model'].lower() == 'deconv_deconv':
            self.reconstruction = self.deconv_deconv_model(self.inv_input)
        elif self.params['inv_model'].lower() == '3_conv_deconv':
            with tf.variable_scope('module_1'):
                one = self.reconstruction = self.conv_deconv_model(self.inv_input)
            with tf.variable_scope('module_2'):
                two = self.reconstruction = self.conv_deconv_model(one)
            with tf.variable_scope('module_3'):
                self.reconstruction = self.conv_deconv_model(two)
        else:
            raise NotImplementedError

        if self.params['channel_losses']:
            reconstruction_channels = tf.split(axis=3, num_or_size_splits=self.reconstruction.get_shape()[3].value,
                                               value=self.reconstruction)
            target_channels = tf.split(axis=3, num_or_size_splits=self.inv_target_channels, value=self.inv_target)
            self.channel_losses = []
            for idx, (rec, tgt) in enumerate(zip(reconstruction_channels, target_channels)):
                self.channel_losses.append(tf.losses.mean_squared_error(tgt, rec))

            self.channel_losses_tensor = tf.stack(axis=0, values=self.channel_losses)
            self.loss = tf.reduce_mean(self.channel_losses_tensor, axis=0)
        else:
            self.loss = tf.losses.mean_squared_error(self.inv_target, self.reconstruction)

        if self.params['optimizer'].lower() == 'adam':
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.params['learning_rate']).minimize(self.loss)
        else:
            raise NotImplementedError

    def conv_deconv_model(self, in_tensor):
        conv_filter = tf.get_variable('conv_filter', shape=[self.params['op1_height'], self.params['op1_width'],
                                                            self.inv_input_channels, self.params['hidden_channels']])

        conv = tf.nn.conv2d(in_tensor, filter=conv_filter, strides=self.params['op1_strides'], padding='SAME')

        conv_bias = tf.get_variable('conv_bias', shape=[self.params['hidden_channels']])
        biased_conv = tf.nn.bias_add(conv, conv_bias)

        relu = tf.nn.relu(biased_conv)

        deconv_filter = tf.get_variable('deconv_filter',
                                        shape=[self.params['op2_height'], self.params['op2_width'],
                                               self.inv_target_channels, self.params['hidden_channels']])
        deconv = tf.nn.conv2d_transpose(relu, filter=deconv_filter,
                                        output_shape=[self.params['batch_size'], self.inv_target_height,
                                                      self.inv_target_width, self.inv_target_channels],
                                        strides=self.params['op2_strides'], padding='SAME')

        deconv_bias = tf.get_variable('deconv_bias', shape=[self.inv_target_channels])
        reconstruction = tf.nn.bias_add(deconv, deconv_bias)
        return reconstruction

    def deconv_conv_model(self, in_tensor):
        # assume for now that the hidden layer has the same dims as the output
        deconv_filter = tf.get_variable('deconv_filter',
                                        shape=[self.params['op1_height'], self.params['op1_width'],
                                               self.params['hidden_channels'], self.inv_input_channels])
        deconv = tf.nn.conv2d_transpose(in_tensor, filter=deconv_filter,
                                        output_shape=[self.params['batch_size'], self.inv_target_height,
                                                      self.inv_target_width, self.params['hidden_channels']],
                                        strides=self.params['op1_strides'], padding='SAME')
        deconv_bias = tf.get_variable('deconv_bias', shape=[self.params['hidden_channels']])
        biased_conv = tf.nn.bias_add(deconv, deconv_bias)

        relu = tf.nn.relu(biased_conv)

        conv_filter = tf.get_variable('conv_filter', shape=[self.params['op2_height'], self.params['op2_width'],
                                                            self.params['hidden_channels'], self.inv_target_channels])

        conv = tf.nn.conv2d(relu, filter=conv_filter, strides=self.params['op2_strides'], padding='SAME')

        conv_bias = tf.get_variable('conv_bias', shape=[self.inv_target_channels])
        reconstruction = tf.nn.bias_add(conv, conv_bias)
        return reconstruction

    def deconv_deconv_model(self, in_tensor):
        deconv_filter_1 = tf.get_variable('deconv_filter_1',
                                          shape=[self.params['op1_height'], self.params['op1_width'],
                                                 self.params['hidden_channels'], self.inv_input_channels])
        deconv_1 = tf.nn.conv2d_transpose(in_tensor, filter=deconv_filter_1,
                                          output_shape=[self.params['batch_size'], self.inv_target_height,
                                                        self.inv_target_width, self.params['hidden_channels']],
                                          strides=self.params['op1_strides'], padding='SAME')
        deconv_bias_1 = tf.get_variable('deconv_bias', shape=[self.params['hidden_channels']])
        biased_deconv_1 = tf.nn.bias_add(deconv_1, deconv_bias_1)

        relu = tf.nn.relu(biased_deconv_1)

        deconv_filter_2 = tf.get_variable('deconv_filter_2',
                                          shape=[self.params['op2_height'], self.params['op2_width'],
                                                 self.inv_target_channels, self.params['hidden_channels']])
        deconv_2 = tf.nn.conv2d_transpose(relu, filter=deconv_filter_2,
                                          output_shape=[self.params['batch_size'], self.inv_target_height,
                                                        self.inv_target_width, self.inv_target_channels],
                                          strides=self.params['op2_strides'], padding='SAME')
        deconv_bias_2 = tf.get_variable('deconv_bias_2', shape=[self.inv_target_channels])
        reconstruction = tf.nn.bias_add(deconv_2, deconv_bias_2)
        return reconstruction

    def build_logging(self, graph):
        tf.summary.scalar('mse_loss', self.loss)
        if self.params['channel_losses']:
            for idx, c_loss in enumerate(self.channel_losses):
                tf.summary.scalar('loss_channel_' + str(idx), c_loss)
            tf.summary.histogram('channel_losses', self.channel_losses_tensor)
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.params['log_path'] + '/summaries', graph)
        self.saver = tf.train.Saver()

        if self.params['test_freq'] > 0:
            self.val_loss = tf.placeholder(dtype=tf.float32, shape=[], name='val_loss')
            self.val_loss_log = tf.summary.scalar('validation_loss', self.val_loss)

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
                img_pl = tf.placeholder(dtype=tf.float32,
                                        shape=[self.params['batch_size'], self.img_hw, self.img_hw, self.img_channels])
                self.build_model(img_pl)
                self.build_logging(sess.graph)
                sess.run(tf.global_variables_initializer())
                start_time = time.time()
                for count in range(self.params['num_iterations']):
                    feed_dict = {img_pl: next(batch_gen)}
                    batch_loss, _, summary_string = sess.run([self.loss, self.train_op, self.summary_op],
                                                             feed_dict=feed_dict)
                    self.summary_writer.add_summary(summary_string, count)

                    if (count + 1) % self.params['print_freq'] == 0:
                        self.summary_writer.flush()
                        # print('Iteration: ' + str(count + 1) +
                        #       ' Train Error: ' + str(batch_loss) +
                        #       ' Time: ' + str((time.time() - start_time) / 60) + 'min')
                        print(('Iteration: {0:6d} Train Error: {1:8.3f} ' +
                               'Time: {2:5.1f} min').format(count + 1, batch_loss, (time.time() - start_time) / 60))

                    if (count + 1) % self.params['log_freq'] == 0 or (count + 1) == self.params['num_iterations']:
                        checkpoint_file = os.path.join(self.params['log_path'], 'ckpt')
                        self.saver.save(sess, checkpoint_file, global_step=(count + 1))

                    if self.params['test_freq'] > 0 and ((count + 1) % self.params['test_freq'] == 0 or
                                                      (count + 1) == self.params['num_iterations']):
                        val_batch_gen = self.get_batch_generator(mode='validate')
                        val_loss_acc = 0.0
                        num_runs = self.params['test_set_size'] // self.params['batch_size'] + 1
                        for val_count in range(num_runs):
                            val_feed_dict = {img_pl: next(val_batch_gen)}
                            val_batch_loss = sess.run(self.loss, feed_dict=val_feed_dict)
                            val_loss_acc += val_batch_loss
                        val_loss_acc /= num_runs
                        val_summary_string = sess.run(self.val_loss_log, feed_dict={self.val_loss: val_loss_acc})
                        self.summary_writer.add_summary(val_summary_string, count)
                        print(('Iteration: {0:6d} Validation Error: {1:8.3f} ' +
                               'Time: {2:5.1f} min').format(count + 1, val_loss_acc, (time.time() - start_time) / 60))

    def run_inverse_model(self, inv_input):
        with tf.Graph().as_default() as graph:
            with tf.Session() as sess:
                if self.params['classifier'].lower() == 'vgg16':
                    model = vgg16.Vgg16()
                elif self.params['classifier'].lower() == 'alexnet':
                    model = alexnet.AlexNet()
                else:
                    raise NotImplementedError

                img_pl = tf.placeholder(dtype=tf.float32,
                                        shape=[self.params['batch_size'], self.img_hw, self.img_hw, self.img_channels])
                model.build(img_pl)
                input_shape = graph.get_tensor_by_name(self.params['inv_input_name']).get_shape()
                self.inv_input = tf.placeholder(dtype=tf.float32, name='inv_input', shape=input_shape)
                self.inv_target = graph.get_tensor_by_name(self.params['inv_target_name'])
                self.inv_input_height = self.inv_input.get_shape()[1].value
                self.inv_input_width = self.inv_input.get_shape()[2].value
                self.inv_input_channels = self.inv_input.get_shape()[3].value
                self.inv_target_height = self.inv_target.get_shape()[1].value
                self.inv_target_width = self.inv_target.get_shape()[2].value
                self.inv_target_channels = self.inv_target.get_shape()[3].value

                if self.params['inv_model'].lower() == 'conv_deconv':
                    self.reconstruction = self.conv_deconv_model(self.inv_input)
                elif self.params['inv_model'].lower() == 'deconv_conv':
                    self.reconstruction = self.deconv_conv_model(self.inv_input)
                elif self.params['inv_model'].lower() == 'deconv_deconv':
                    self.reconstruction = self.deconv_deconv_model(self.inv_input)
                elif self.params['inv_model'].lower() == '3_conv_deconv':
                    with tf.variable_scope('module_1'):
                        one = self.reconstruction = self.conv_deconv_model(self.inv_input)
                    with tf.variable_scope('module_2'):
                        two = self.reconstruction = self.conv_deconv_model(one)
                    with tf.variable_scope('module_3'):
                        self.reconstruction = self.conv_deconv_model(two)
                else:
                    raise NotImplementedError

                saver = tf.train.Saver()
                saver.restore(sess, self.params['load_path'])

                rec_mat = sess.run(self.reconstruction, feed_dict={self.inv_input: inv_input})
                return rec_mat

    def visualize(self, num_images=5, rec_type='rgb_scaled', file_name='img_vs_rec', add_diffs=True):
        actual_batch_size = self.params['batch_size']
        assert num_images <= actual_batch_size
        self.params['batch_size'] = num_images

        batch_gen = self.get_batch_generator(mode='validate')

        with tf.Graph().as_default():
            with tf.Session() as sess:
                img_pl = tf.placeholder(dtype=tf.float32,
                                        shape=[self.params['batch_size'], self.img_hw, self.img_hw, self.img_channels])
                self.build_model(img_pl)
                saver = tf.train.Saver()
                saver.restore(sess, self.params['load_path'])
                feed_dict = {img_pl: next(batch_gen)}
                reconstruction = sess.run([self.reconstruction], feed_dict=feed_dict)

        self.params['batch_size'] = actual_batch_size

        img_mat = feed_dict[img_pl][:num_images, :, :, :]
        rec_mat = reconstruction[0][:num_images, :, :, :]

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

    def visualize_old(self, img_idx=0, rec_type='rgb_scaled'):
        batch_gen = self.get_batch_generator(mode='validate')

        with tf.Graph().as_default() as graph:
            with tf.Session() as sess:
                img_pl = tf.placeholder(dtype=tf.float32,
                                        shape=[self.params['batch_size'], self.img_hw, self.img_hw, self.img_channels])
                self.build_model(img_pl)
                saver = tf.train.Saver()
                saver.restore(sess, self.params['load_path'])
                feed_dict = {img_pl: next(batch_gen)}
                reconstruction = sess.run([self.reconstruction], feed_dict=feed_dict)

            img_mat = feed_dict[img_pl][img_idx, :, :, :]
            rec_mat = reconstruction[0][img_idx, :, :, :]
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
    params_list = [p._replace(batch_size=num_images) for p in params_list]
    li = LayerInversion(params_list[0])

    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:
            if params_list[0]['classifier'].lower() == 'vgg16':
                model = vgg16.Vgg16()
            elif params_list[0]['classifier'].lower() == 'alexnet':
                model = alexnet.AlexNet()
            else:
                raise NotImplementedError
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
        li = LayerInversion(params)
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
