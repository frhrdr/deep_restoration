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

Parameters = namedtuple('Paramters', ['conv1_height', 'conv1_width',
                                      'conv2_height', 'conv2_width', 'conv2_channels',
                                      'learning_rate', 'batch_size', 'num_iterations',
                                      'optimizer',
                                      'data_path', 'images_file',
                                      'log_path', 'load_path',
                                      'log_freq', 'test_freq'])

params = Parameters(conv1_height=5, conv1_width=5,
                    conv2_height=5, conv2_width=5, conv2_channels=64,
                    learning_rate=0.00001, batch_size=10, num_iterations=1000,
                    optimizer=tf.train.AdamOptimizer,
                    data_path='./data/imagenet2012-validationset/', images_file='val_images.txt',
                    log_path='./logs/vgg_inversion_layer_1/run1/',
                    load_path='./logs/vgg_inversion_layer_1/run1/ckpt-500',
                    log_freq=500, test_freq=-1)


class VggLayer1Inversion:

    def __init__(self, params):
        self.params = params
        self.img_channels = 3
        self.feat_channels = 64
        self.vgg_mean = [103.939, 116.779, 123.68]

    def build_model(self, img_pl):
        vgg = vgg16.Vgg16()
        vgg.build(img_pl)
        self.feat1 = vgg.conv1_1

        self.filter1 = tf.get_variable('filter1', shape=[self.params.conv1_height, self.params.conv1_width,
                                                         self.feat_channels, self.params.conv2_channels])
        self.conv1 = tf.nn.conv2d(self.feat1, filter=self.filter1, strides=[1, 1, 1, 1], padding='SAME')
        self.bias1 = tf.get_variable('bias1', shape=[self.params.conv2_channels])
        self.bconv1 = tf.nn.bias_add(self.conv1, self.bias1)

        self.relu = tf.nn.relu(self.bconv1)

        self.filter2 = tf.get_variable('filter2', shape=[self.params.conv2_height, self.params.conv2_width,
                                                         self.params.conv2_channels, self.img_channels])
        self.conv2 = tf.nn.conv2d(self.relu, filter=self.filter2, strides=[1, 1, 1, 1], padding='SAME')
        self.bias2 = tf.get_variable('bias2', shape=[self.img_channels])
        self.rec = tf.nn.bias_add(self.conv2, self.bias2)

        self.loss = tf.losses.mean_squared_error(img_pl, self.rec - self.vgg_mean)

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

            batch_names = [k.split('.')[0] for k in batch_files]
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

        with tf.Session() as sess:
            img_pl = tf.placeholder(dtype=tf.float32, shape=[params.batch_size, 224, 224, 3])
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

        with tf.Session() as sess:
            img_pl = tf.placeholder(dtype=tf.float32, shape=[params.batch_size, 224, 224, 3])
            self.build_model(img_pl)
            saver = tf.train.Saver()
            saver.restore(sess, self.params.load_path)
            feed_dict = {img_pl: next(batch_gen)}
            reconstruction = sess.run([self.rec], feed_dict=feed_dict)

        idx = 0
        img_mat = feed_dict[img_pl][idx, :, :, :]
        rec_mat = reconstruction[0][idx, :, :, :] + self.vgg_mean

        plt.imshow(img_mat)
        plt.savefig(self.params.log_path + 'img' + str(idx) + '.png', format='png')
        plt.figure()
        plt.imshow(rec_mat)
        plt.savefig(self.params.log_path + 'rec' + str(idx) + '.png', format='png')

VggLayer1Inversion(params).train()