import tensorflow as tf
from tensorflow_vgg import vgg16
from tensorflow_vgg import utils as vgg_utils
from skimage.color import grey2rgb
import os
import numpy as np


def get_batchwise_iterator(batch_size, im_file, data_path="./data/imagenet2012-validationset/"):
    with open(im_file) as f:
        image_files = [k.rstrip() for k in f.readlines()]

    assert len(image_files) % batch_size == 0

    while image_files:
        batch_files = image_files[:batch_size]
        image_files = image_files[batch_size:]
        batch_names = [k.split('.')[0] for k in batch_files]
        batch_paths = [data_path + 'images/' + k for k in batch_files]
        images = []
        for ipath in batch_paths:
            image = vgg_utils.load_image(ipath)
            if len(image.shape) == 2:
                image = grey2rgb(image)
            images.append(image)
        mat = np.stack(images, axis=0)
        yield mat, batch_names


def vgg_layer_dict(vgg):
    layers = {
        "conv1_1": vgg.conv1_1, "conv1_2": vgg.conv1_2, "pool1": vgg.pool1,
        "conv2_1": vgg.conv2_1, "conv2_2": vgg.conv2_2, "pool2": vgg.pool2,
        "conv3_1": vgg.conv3_1, "conv3_2": vgg.conv3_2, "conv3_3": vgg.conv3_3, "pool3": vgg.pool3,
        "conv4_1": vgg.conv4_1, "conv4_2": vgg.conv4_2, "conv4_3": vgg.conv4_3, "pool4": vgg.pool4,
        "conv5_1": vgg.conv5_1, "conv5_2": vgg.conv5_2, "conv5_3": vgg.conv5_3, "pool5": vgg.pool5,
        "fc6": vgg.fc6, "relu6": vgg.relu6,
        "fc7": vgg.fc7, "relu7": vgg.relu7,
        "fc8": vgg.fc8, "prob": vgg.prob}

    return layers


def get_layer_features(layer, batch_size, im_file='subset_cutoff_200_images.txt', data_path="./data/imagenet2012-validationset/"):
    if not os.path.exists('./data/features/' + layer):
        os.makedirs('./data/features/' + layer)

    batches = get_batchwise_iterator(batch_size, data_path + im_file)

    with tf.Session() as sess:
        images = tf.placeholder("float", [batch_size, 224, 224, 3])
        vgg = vgg16.Vgg16()

        with tf.name_scope("content_vgg"):
            vgg.build(images)

        tracked_layer = vgg_layer_dict(vgg)[layer]

        for batch, names in batches:
            batch_features = sess.run(tracked_layer, feed_dict={images: batch})
            for idx, name in enumerate(names):
                if len(batch_features.shape) == 4:
                    feature_map = batch_features[idx, :, :, :]
                elif len(batch_features.shape) == 2:
                    feature_map = batch_features[idx, :]
                else:
                    raise NotImplementedError

                np.save('./data/features/' + layer + '/' + layer + '_' + name + '.npy', feature_map)


def test_run(data_path="./data/imagenet2012-validationset/"):
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
