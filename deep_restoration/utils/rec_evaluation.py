import os.path

import numpy as np
import tensorflow as tf

from modules.loss_modules import VggScoreLoss, MSELoss, NormedMSELoss
from utils.filehandling import load_image


def subset10_paths(classifier):
    _, img_hw, _ = classifier_stats(classifier)
    subset_dir = '../data/selected/images_resized_{}/'.format(img_hw)
    img_ids = selected_img_ids()
    subset_paths = ['{}val{}.bmp'.format(subset_dir, i) for i in img_ids]
    return subset_paths


def selected_img_ids():
    # alexnet top1 correct 53, 76, 81, 129, 160
    # vgg16 top1 correct
    return 53, 76, 81, 99, 106, 108, 129, 153, 157, 160


def mv_mse_and_vgg_scores(classifier, img_based=True):
    tgt_paths = subset10_paths(classifier)
    _, img_hw, layer_names = classifier_stats(classifier)
    log_path = '../logs/mahendran_vedaldi/2016/{}/'.format(classifier)
    layer_subdirs = [l.replace('/', '_') for l in layer_names]
    img_subdirs = [p.split('/')[-1].split('.')[0] for p in tgt_paths]

    tgt_images = [np.expand_dims(load_image(p), axis=0) for p in tgt_paths]
    rec_filename = 'imgs/rec_3500.png' if img_based else 'mats/rec_3500.npy'

    vgg_loss = VggScoreLoss(('tgt_224:0', 'rec_224:0'), weighting=1.0, name=None, input_scaling=1.0)
    mse_loss = MSELoss('tgt_pl:0', 'rec_pl:0')
    nmse_loss = NormedMSELoss('tgt_pl:0', 'rec_pl:0')
    loss_mods = [vgg_loss, mse_loss, nmse_loss]

    found_layers = []
    score_list = []

    with tf.Graph().as_default():
        tgt_pl = tf.placeholder(dtype=tf.float32, shape=(1, img_hw, img_hw, 3), name='tgt_pl')
        rec_pl = tf.placeholder(dtype=tf.float32, shape=(1, img_hw, img_hw, 3), name='rec_pl')
        _ = tf.slice(tgt_pl, begin=[0, 0, 0, 0], size=[-1, 224, 224, -1], name='tgt_224')
        _ = tf.slice(rec_pl, begin=[0, 0, 0, 0], size=[-1, 224, 224, -1], name='rec_224')

        for lmod in loss_mods:
            lmod.build()
        loss_tsr_list = [m.get_loss() for m in loss_mods]

        with tf.Session() as sess:

            for layer_subdir in layer_subdirs:
                layer_log_path = '{}{}/'.format(log_path, layer_subdir)
                if not os.path.exists(layer_log_path):
                    continue
                found_layers.append(layer_subdir)
                layer_score_list = []

                for idx, img_subdir in enumerate(img_subdirs):
                    img_log_path = '{}{}/'.format(layer_log_path, img_subdir)
                    rec_image = np.expand_dims(load_image(img_log_path + rec_filename), axis=0)
                    if np.max(rec_image) < 2.:
                        rec_image = rec_image * 255.
                    scores = sess.run(loss_tsr_list, feed_dict={tgt_pl: tgt_images[idx], rec_pl: rec_image})
                    layer_score_list.append(scores)
                score_list.append(layer_score_list)

    score_mat = np.asarray(score_list)
    print(score_mat.shape)
    print(found_layers)
    np.save('{}score_mat.npy'.format(log_path), score_mat)
    return score_mat


def classifier_stats(classifier):
    assert classifier in ('alexnet', 'vgg16')
    if classifier == 'alexnet':
        imagenet_mean = (123.68 + 116.779 + 103.939) / 3
        img_hw = 227
        layers = ['conv1/lin', 'conv1/relu', 'lrn1', 'pool1',
                  'conv2/lin', 'conv2/relu', 'lrn2', 'pool2',
                  'conv3/lin', 'conv3/relu', 'conv4/lin', 'conv4/relu', 'conv5/lin', 'conv5/relu', 'pool5',
                  'fc6/lin', 'fc6/relu', 'fc7/lin', 'fc7/relu', 'fc8/lin', 'fc8/relu', 'softmax']
    else:
        imagenet_mean = [123.68, 116.779, 103.939]
        img_hw = 224
        layers = ['conv1_1/lin', 'conv1_1/relu', 'conv1_2/lin', 'conv1_2/relu', 'pool1',
                  'conv2_1/lin', 'conv2_1/relu', 'conv2_2/lin', 'conv2_2/relu', 'pool2',
                  'conv3_1/lin', 'conv3_1/relu', 'conv3_2/lin', 'conv3_2/relu', 'conv3_3/lin', 'conv3_3/relu', 'pool3',
                  'conv4_1/lin', 'conv4_1/relu', 'conv4_2/lin', 'conv4_2/relu', 'conv4_3/lin', 'conv4_3/relu', 'pool4',
                  'conv5_1/lin', 'conv5_1/relu', 'conv5_2/lin', 'conv5_2/relu', 'conv5_3/lin', 'conv5_3/relu', 'pool5',
                  'fc6/lin', 'fc6/relu', 'fc7/lin', 'fc7/relu', 'fc8/lin', 'fc8/relu', 'softmax']
    return imagenet_mean, img_hw, layers