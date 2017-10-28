import os.path

import numpy as np
import skimage.io
import tensorflow as tf

from modules.inv_default_modules import get_stacked_module
from modules.loss_modules import NormedMSELoss, MSELoss, VggScoreLoss
from net_inversion import NetInversion
from utils.filehandling import load_image


def selected_img_ids():
    # alexnet top1 correct 53, 76, 81, 129, 160
    # vgg16 top1 correct
    return 53, 76, 81, 99, 106, 108, 129, 153, 157, 160


def subset10_paths(classifier):
    _, img_hw, _ = classifier_stats(classifier)
    subset_dir = '../data/selected/images_resized_{}/'.format(img_hw)
    img_ids = selected_img_ids()
    subset_paths = ['{}val{}.bmp'.format(subset_dir, i) for i in img_ids]
    return subset_paths


def load_and_stack_imgs(img_paths):
    return np.stack([load_image(p) for p in img_paths], axis=0)


def cnn_inv_log_path(classifier, start_layer, rec_layer):
    return '../logs/cnn_inversion/{}/stack_{}_to_{}/'.format(classifier, start_layer, rec_layer)


def run_stacked_module(classifier, start_layer, rec_layer, use_solotrain=False,
                       subdir_name=None, retrieve_special=None):

    subdir_name = subdir_name or '{}_stack_{}_to_{}'.format(classifier, start_layer, rec_layer)
    alt_load_subdir = 'solotrain' if use_solotrain else subdir_name
    module_list = get_stacked_module(classifier, start_layer, rec_layer, alt_load_subdir=alt_load_subdir,
                                     subdir_name=subdir_name, trainable=False)
    save_subdir = 'stacked/' if use_solotrain else 'merged/'
    log_path = cnn_inv_log_path(classifier, start_layer, rec_layer) + save_subdir
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    ni = NetInversion(module_list, log_path, classifier=classifier)
    img_paths = subset10_paths(classifier)
    img_mat = load_and_stack_imgs(img_paths)
    to_fetch = retrieve_special or ('{}/{}:0'.format(module_list[-1].name, module_list[-1].rec_name),)
    recs = ni.run_model_on_images(img_mat, to_fetch)
    for name, rec in zip(to_fetch, recs):
        print(rec.shape)
        np.save('{}cnn_rec_{}.npy'.format(log_path, name.replace('/', '_')), rec)

    if retrieve_special is None:
        images = [np.squeeze(k, axis=0) for k in np.split(recs[0], indices_or_sections=recs[0].shape[0], axis=0)]
        images = [np.minimum(np.maximum(k, 0.), 255.) / 255. for k in images]
        image_ids = [p.split('/')[-1].split('.')[0][len('val'):] for p in img_paths]
        image_save_paths = [log_path + 'img_rec_{}.png'.format(i) for i in image_ids]
        for path, img in zip(image_save_paths, images):
            skimage.io.imsave(path, img)


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


def db_img_mse_and_vgg_scores(classifier):
    tgt_paths = subset10_paths(classifier)
    _, img_hw, layer_names = classifier_stats(classifier)
    tgt_images = [load_image(p) for p in tgt_paths]
    rec_filenames = ['img_rec_{}.png'.format(i) for i in selected_img_ids()]
    save_subdirs = ('stacked/', 'merged/')
    start_layers = list(range(2, 11))

    tgt_pl = tf.placeholder(dtype=tf.float32, shape=(1, img_hw, img_hw, 3))
    rec_pl = tf.placeholder(dtype=tf.float32, shape=(1, img_hw, img_hw, 3))

    vgg_loss = VggScoreLoss((tgt_pl, rec_pl), weighting=1.0, name=None, input_scaling=1.0)
    mse_loss = MSELoss(tgt_pl, rec_pl)
    nmse_loss = NormedMSELoss(tgt_pl, rec_pl)
    loss_mods = [vgg_loss, mse_loss, nmse_loss]

    found_layers = []
    score_list = []

    with tf.Graph().as_default():
        for lmod in loss_mods:
            lmod.build()
        loss_tsr_list = [m.get_loss() for m in loss_mods]

        with tf.Session() as sess:
            for start_layer in start_layers:
                layer_list = []
                for save_subdir in save_subdirs:
                    load_path = cnn_inv_log_path(classifier, start_layer, rec_layer=1) + save_subdir
                    if not os.path.exists(load_path):
                        continue
                    save_subdir_list = []
                    for idx, rec_filename in enumerate(rec_filenames):

                        rec_image = load_image(load_path + rec_filename)
                        scores = sess.run(loss_tsr_list, feed_dict={tgt_pl: tgt_images[idx], rec_pl: rec_image})
                        save_subdir_list.append(scores)
                    layer_list.append(save_subdir_list)
                score_list.append(layer_list)

    score_mat = np.asarray(score_list)
    print(score_mat.shape)
    print(found_layers)
    np.save('../logs/cnn_inversion/{}/score_mat.npy'.format(classifier), score_mat)


def db_lin_to_lin_mse_scores(classifier, use_solotrain=False):
    lin_start_rec_pairs = ((4, 2), (7, 5), (8, 8), (9, 9))
    lin_tensor_names = [('DC{}/c{}l_rec:0'.format(i + 1, lin_start_rec_pairs[i][1]),) for i in range(4)]
    for lin_tensor, pair in zip(lin_tensor_names, lin_start_rec_pairs):
        start_layer, rec_layer = pair
        run_stacked_module(classifier, start_layer, rec_layer, use_solotrain=use_solotrain,
                           subdir_name=None, retrieve_special=lin_tensor)


def db_lin_to_img_gen(classifier, use_solotrain=False):
    lin_start_rec_pairs = ((4, 1), (7, 1), (8, 1), (9, 1))
    lin_tensor_name = ('DC1/rgb_rec:0',)
    for start_layer, rec_layer in lin_start_rec_pairs:

        run_stacked_module(classifier, start_layer, rec_layer, use_solotrain=use_solotrain,
                           subdir_name=None, retrieve_special=lin_tensor_name)


