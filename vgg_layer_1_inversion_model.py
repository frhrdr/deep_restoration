import tensorflow as tf
from tensorflow_vgg import vgg16
from tensorflow_vgg import utils as vgg_utils
from collections import namedtuple
import os
import time
import numpy as np
import matplotlib
matplotlib.use('qt5agg', warn=False, force=True)
import matplotlib.pyplot as plt
from skimage.color import grey2rgb

Parameters = namedtuple('Paramters', ['conv_height', 'conv_width',
                                      'deconv_height', 'deconv_width', 'deconv_channels',
                                      'learning_rate', 'batch_size', 'num_iterations',
                                      'optimizer',
                                      'data_path', 'images_file',
                                      'log_path', 'load_path',
                                      'log_freq', 'test_freq'])


class VggLayer1Inversion:

    def __init__(self, params):
        self.params = params
        self.img_channels = 3
        self.img_hw = 224
        self.feat_channels = 64
        self.vgg_mean = np.asarray([103.939, 116.779, 123.68])

    def build_model(self, img_pl):
        vgg = vgg16.Vgg16()
        vgg.build(img_pl)
        self.layer1_feat = vgg.conv1_1

        self.conv_filter = tf.get_variable('conv_filter', shape=[self.params.conv_height, self.params.conv_width,
                                                             self.feat_channels, self.params.deconv_channels])
        self.conv = tf.nn.conv2d(self.layer1_feat, filter=self.conv_filter, strides=[1, 1, 1, 1], padding='SAME')
        self.conv_bias = tf.get_variable('conv_bias', shape=[self.params.deconv_channels])
        self.biased_conv = tf.nn.bias_add(self.conv, self.conv_bias)

        self.relu = tf.nn.relu(self.biased_conv)

        self.deconv_filter = tf.get_variable('deconv_filter', shape=[self.params.deconv_height, self.params.deconv_width,
                                                               self.img_channels, self.params.deconv_channels])
        self.deconv = tf.nn.conv2d_transpose(self.relu, filter=self.deconv_filter,
                                             output_shape=[self.params.batch_size, self.img_hw, self.img_hw,
                                                           self.img_channels],
                                             strides=[1, 1, 1, 1], padding='SAME')
        self.deconv_bias = tf.get_variable('deconv_bias', shape=[self.img_channels])
        self.reconstruction = tf.nn.bias_add(self.deconv, self.deconv_bias)

        self.loss = tf.losses.mean_squared_error(img_pl * 255.0 - self.vgg_mean, self.reconstruction)

        self.train_op = self.params.optimizer(learning_rate=self.params.learning_rate).minimize(self.loss)

    def build_logging(self, graph):
        tf.summary.scalar('mse_loss', self.loss)
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.params.log_path + '/summaries', graph)
        self.saver = tf.train.Saver()

    def get_batch_generator(self):
        with open(self.params.data_path + self.params.images_file) as f:
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
                image = vgg_utils.load_image(img_path)
                if len(image.shape) == 2:
                    image = grey2rgb(image)
                images.append(image)
            mat = np.stack(images, axis=0)
            yield mat

    def train(self):
        if not os.path.exists(self.params.log_path):
            os.makedirs(self.params.log_path)

        batch_gen = self.get_batch_generator()
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

                    if (count + 1) % 10 == 0:
                        self.summary_writer.flush()
                        print('Iteration: ' + str(count + 1) +
                              ' Train Error: ' + str(batch_loss) +
                              ' Time: ' + str(time.time() - start_time))

                    if (count + 1) % self.params.log_freq == 0 or (count + 1) == self.params.num_iterations:
                        checkpoint_file = os.path.join(self.params.log_path, 'ckpt')
                        self.saver.save(sess, checkpoint_file, global_step=(count + 1))

    def visualize(self):
        batch_gen = self.get_batch_generator()

        with tf.Graph().as_default():
            with tf.Session() as sess:
                img_pl = tf.placeholder(dtype=tf.float32,
                                        shape=[self.params.batch_size, self.img_hw, self.img_hw, self.img_channels])
                self.build_model(img_pl)
                saver = tf.train.Saver()
                saver.restore(sess, self.params.load_path)
                feed_dict = {img_pl: next(batch_gen)}
                reconstruction = sess.run([self.reconstruction], feed_dict=feed_dict)

            idx = 0
            img_mat = feed_dict[img_pl][idx, :, :, :]
            rec_mat = reconstruction[0][idx, :, :, :] + np.array(self.vgg_mean)
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
            plt.savefig(self.params.log_path + 'img' + str(idx) + '.png', dpi=224 / 4, format='png')
            fig = plt.figure(frameon=False)
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(rec_mat, aspect='auto')
            plt.savefig(self.params.log_path + 'rec' + str(idx) + '.png', dpi=224 / 4, format='png')


# params = Parameters(conv_height=1, conv_width=1,
#                     deconv_height=3, deconv_width=3, deconv_channels=64,
#                     learning_rate=0.0001, batch_size=10, num_iterations=300,
#                     optimizer=tf.train.AdamOptimizer,
#                     data_path='./data/imagenet2012-validationset/', images_file='val_images.txt',
#                     log_path='./logs/vgg_inversion_layer_1/run3/',
#                     load_path='./logs/vgg_inversion_layer_1/run3/ckpt-300',
#                     log_freq=1000, test_freq=-1)
#
# VggLayer1Inversion(params).train()
# VggLayer1Inversion(params).visualize()
