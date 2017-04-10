import tensorflow as tf
from tensorflow_vgg import vgg16
from tensorflow_vgg import utils as vgg_utils

import numpy as np

data_path = "./data/imagenet2012-validationset/"

img1 = vgg_utils.load_image(data_path + "images/ILSVRC2012_val_00000001.JPEG")
img2 = vgg_utils.load_image(data_path + "images/ILSVRC2012_val_00000002.JPEG")

batch1 = img1.reshape((1, 224, 224, 3))
batch2 = img2.reshape((1, 224, 224, 3))

batch = np.concatenate((batch1, batch2), 0)

with tf.Session() as sess:
    images = tf.placeholder("float", [2, 224, 224, 3])
    feed_dict = {images: batch}

    vgg = vgg16.Vgg16()
    with tf.name_scope("content_vgg"):
        vgg.build(images)

    prob, c1_2 = sess.run([vgg.prob, vgg.conv1_2], feed_dict=feed_dict)
    vgg_utils.print_prob(prob[0], data_path + 'label_names.txt')
    vgg_utils.print_prob(prob[1], data_path + 'label_names.txt')