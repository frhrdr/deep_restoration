import tensorflow as tf
from tf_vgg import vgg16
from tf_alexnet import alexnet
import filehandling_utils
from collections import namedtuple
import os
import time
import numpy as np
import matplotlib
matplotlib.use('qt5agg', warn=False, force=True)
import matplotlib.pyplot as plt
from skimage.color import grey2rgb

Parameters = namedtuple('Paramters', ['classifier', 'inv_input_name', 'inv_target_name',
                                      'inv_model',
                                      'op1_height', 'op1_width', 'op1_strides',
                                      'op2_height', 'op2_width', 'op2_strides',
                                      'hidden_channels',
                                      'learning_rate', 'batch_size', 'num_iterations',
                                      'optimizer',
                                      'data_path', 'train_images_file', 'validation_images_file',
                                      'log_path', 'load_path',
                                      'print_freq', 'log_freq', 'test_freq',
                                      'test_set_size'])


class LayerInversion:

    def __init__(self, params):
        self.params = params
        self.imagenet_mean = np.asarray([123.68, 116.779, 103.939])  # in RGB order
        self.img_hw = 224
        self.img_channels = 3

    def build_model(self, img_pl):
        if self.params.classifier.lower() == 'vgg16':
            model = vgg16.Vgg16()
        elif self.params.classifier.lower() == 'alexnet':
            model = alexnet.AlexNet()
        else:
            raise NotImplementedError

        model.build(img_pl)
        graph = tf.get_default_graph()

        self.inv_input = graph.get_tensor_by_name(self.params.inv_input_name)
        self.inv_target = graph.get_tensor_by_name(self.params.inv_target_name)
        self.inv_input_height = self.inv_input.get_shape()[1].value
        self.inv_input_width = self.inv_input.get_shape()[2].value
        self.inv_input_channels = self.inv_input.get_shape()[3].value
        self.inv_target_height = self.inv_target.get_shape()[1].value
        self.inv_target_width = self.inv_target.get_shape()[2].value
        self.inv_target_channels = self.inv_target.get_shape()[3].value

        if self.params.inv_model.lower() == 'conv_deconv':
            self.reconstruction = self.conv_deconv_model()
        elif self.params.inv_model.lower() == 'deconv_conv':
            self.reconstruction = self.deconv_conv_model()
        elif self.params.inv_model.lower() == 'deconv_deconv':
            self.reconstruction = self.deconv_deconv_model()
        else:
            raise NotImplementedError

        if True:
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

        if self.params.optimizer.lower() == 'adam':
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate).minimize(self.loss)
        else:
            raise NotImplementedError

    def conv_deconv_model(self):
        conv_filter = tf.get_variable('conv_filter', shape=[self.params.op1_height, self.params.op1_width,
                                                            self.inv_input_channels, self.params.hidden_channels])

        conv = tf.nn.conv2d(self.inv_input, filter=conv_filter, strides=self.params.op1_strides, padding='SAME')

        conv_bias = tf.get_variable('conv_bias', shape=[self.params.hidden_channels])
        biased_conv = tf.nn.bias_add(conv, conv_bias)

        relu = tf.nn.relu(biased_conv)

        deconv_filter = tf.get_variable('deconv_filter',
                                        shape=[self.params.op2_height, self.params.op2_width,
                                               self.inv_target_channels, self.params.hidden_channels])
        deconv = tf.nn.conv2d_transpose(relu, filter=deconv_filter,
                                        output_shape=[self.params.batch_size, self.inv_target_height,
                                                      self.inv_target_width, self.inv_target_channels],
                                        strides=self.params.op2_strides, padding='SAME')
        deconv_bias = tf.get_variable('deconv_bias', shape=[self.inv_target_channels])
        reconstruction = tf.nn.bias_add(deconv, deconv_bias)
        return reconstruction

    def deconv_conv_model(self):
        # assume for now that the hidden layer has the same dims as the output
        deconv_filter = tf.get_variable('deconv_filter',
                                        shape=[self.params.op1_height, self.params.op1_width,
                                               self.params.hidden_channels, self.inv_input_channels])
        deconv = tf.nn.conv2d_transpose(self.inv_input, filter=deconv_filter,
                                        output_shape=[self.params.batch_size, self.inv_target_height,
                                                      self.inv_target_width, self.params.hidden_channels],
                                        strides=self.params.op1_strides, padding='SAME')
        deconv_bias = tf.get_variable('deconv_bias', shape=[self.params.hidden_channels])
        biased_conv = tf.nn.bias_add(deconv, deconv_bias)

        relu = tf.nn.relu(biased_conv)

        conv_filter = tf.get_variable('conv_filter', shape=[self.params.op2_height, self.params.op2_width,
                                                            self.params.hidden_channels, self.inv_target_channels])

        conv = tf.nn.conv2d(relu, filter=conv_filter, strides=self.params.op2_strides, padding='SAME')

        conv_bias = tf.get_variable('conv_bias', shape=[self.inv_target_channels])
        reconstruction = tf.nn.bias_add(conv, conv_bias)
        return reconstruction

    def deconv_deconv_model(self):
        deconv_filter_1 = tf.get_variable('deconv_filter_1',
                                          shape=[self.params.op1_height, self.params.op1_width,
                                                 self.params.hidden_channels, self.inv_input_channels])
        deconv_1 = tf.nn.conv2d_transpose(self.inv_input, filter=deconv_filter_1,
                                          output_shape=[self.params.batch_size, self.inv_target_height,
                                                        self.inv_target_width, self.params.hidden_channels],
                                          strides=self.params.op1_strides, padding='SAME')
        deconv_bias_1 = tf.get_variable('deconv_bias', shape=[self.params.hidden_channels])
        biased_deconv_1 = tf.nn.bias_add(deconv_1, deconv_bias_1)

        relu = tf.nn.relu(biased_deconv_1)

        deconv_filter_2 = tf.get_variable('deconv_filter_2',
                                          shape=[self.params.op2_height, self.params.op2_width,
                                                 self.inv_target_channels, self.params.hidden_channels])
        deconv_2 = tf.nn.conv2d_transpose(relu, filter=deconv_filter_2,
                                          output_shape=[self.params.batch_size, self.inv_target_height,
                                                        self.inv_target_width, self.inv_target_channels],
                                          strides=self.params.op2_strides, padding='SAME')
        deconv_bias_2 = tf.get_variable('deconv_bias_2', shape=[self.inv_target_channels])
        reconstruction = tf.nn.bias_add(deconv_2, deconv_bias_2)
        return reconstruction

    def build_logging(self, graph):
        tf.summary.scalar('mse_loss', self.loss)
        for idx, c_loss in enumerate(self.channel_losses):
            tf.summary.scalar('loss_channel_' + str(idx), c_loss)
        tf.summary.histogram('channel_losses', self.channel_losses_tensor)
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.params.log_path + '/summaries', graph)
        self.saver = tf.train.Saver()

        if self.params.test_freq > 0:
            self.val_loss = tf.placeholder(dtype=tf.float32, shape=[], name='val_loss')
            self.val_loss_log = tf.summary.scalar('validation_loss', self.val_loss)

    def get_batch_generator(self, mode='train'):
        if mode == 'train':
            img_file = self.params.train_images_file
        elif mode == 'validate':
            img_file = self.params.validation_images_file
        else:
            raise NotImplementedError

        with open(self.params.data_path + img_file) as f:
            image_files = [k.rstrip() for k in f.readlines()]

        begin = 0
        while True:
            end = begin + self.params.batch_size
            if end < len(image_files):
                batch_files = image_files[begin:end]
            else:
                end = end - len(image_files)
                batch_files = image_files[begin:] + image_files[:end]
            begin = end

            batch_paths = [self.params.data_path + 'images/' + k for k in batch_files]
            images = []
            for img_path in batch_paths:
                image = filehandling_utils.load_image(img_path)
                if len(image.shape) == 2:
                    image = grey2rgb(image)
                images.append(image)
            mat = np.stack(images, axis=0)
            yield mat

    def train(self):
        if not os.path.exists(self.params.log_path):
            os.makedirs(self.params.log_path)

        filehandling_utils.save_namedtuple(self.params, self.params.log_path + 'params.txt')

        batch_gen = self.get_batch_generator(mode='train')
        with tf.Graph().as_default():
            with tf.Session() as sess:
                img_pl = tf.placeholder(dtype=tf.float32,
                                        shape=[self.params.batch_size, self.img_hw, self.img_hw, self.img_channels])
                self.build_model(img_pl)
                self.build_logging(sess.graph)
                sess.run(tf.global_variables_initializer())
                start_time = time.time()
                for count in range(self.params.num_iterations):
                    feed_dict = {img_pl: next(batch_gen)}
                    batch_loss, _, summary_string = sess.run([self.loss, self.train_op, self.summary_op],
                                                             feed_dict=feed_dict)
                    self.summary_writer.add_summary(summary_string, count)

                    if (count + 1) % self.params.print_freq == 0:
                        self.summary_writer.flush()
                        print('Iteration: ' + str(count + 1) +
                              ' Train Error: ' + str(batch_loss) +
                              ' Time: ' + str(time.time() - start_time))

                    if (count + 1) % self.params.log_freq == 0 or (count + 1) == self.params.num_iterations:
                        checkpoint_file = os.path.join(self.params.log_path, 'ckpt')
                        self.saver.save(sess, checkpoint_file, global_step=(count + 1))

                    if self.params.test_freq > 0 and ((count + 1) % self.params.test_freq or
                                                      (count + 1) == self.params.num_iterations):
                        val_batch_gen = self.get_batch_generator(mode='validate')
                        val_loss_acc = 0.0
                        num_runs = self.params.test_set_size // self.params.batch_size + 1
                        for val_count in range(num_runs):
                            val_feed_dict = {img_pl: next(val_batch_gen)}
                            val_batch_loss = sess.run(self.loss, feed_dict=val_feed_dict)
                            val_loss_acc += val_batch_loss
                        val_loss_acc /= num_runs
                        val_summary_string = sess.run(self.val_loss_log, feed_dict={self.val_loss: val_loss_acc})
                        self.summary_writer.add_summary(val_summary_string, count)
                        print('Iteration: ' + str(count + 1) +
                              ' Validation Error: ' + str(val_loss_acc) +
                              ' Time: ' + str(time.time() - start_time))

    def visualize(self, img_idx=0):
        batch_gen = self.get_batch_generator(mode='validate')

        with tf.Graph().as_default() as graph:
            with tf.Session() as sess:
                img_pl = tf.placeholder(dtype=tf.float32,
                                        shape=[self.params.batch_size, self.img_hw, self.img_hw, self.img_channels])
                self.build_model(img_pl)
                saver = tf.train.Saver()
                saver.restore(sess, self.params.load_path)
                feed_dict = {img_pl: next(batch_gen)}
                reconstruction = sess.run([self.reconstruction], feed_dict=feed_dict)

            img_mat = feed_dict[img_pl][img_idx, :, :, :]
            rec_mat = reconstruction[0][img_idx, :, :, :]  # + np.array(self.imagenet_mean)
            rec_mat /= 255.0
            print('reconstruction min and max vals: ' + str(rec_mat.min()) + ', ' + str(rec_mat.max()))
            rec_mat = np.minimum(np.maximum(rec_mat, 0.0), 1.0)

            w = h = 4
            fig = plt.figure(frameon=False)
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(img_mat, aspect='auto')
            plt.savefig(self.params.log_path + 'img' + str(img_idx) + '.png', dpi=224 / 4, format='png')
            fig = plt.figure(frameon=False)
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(rec_mat, aspect='auto')
            plt.savefig(self.params.log_path + 'rec' + str(img_idx) + '.png', dpi=224 / 4, format='png')

