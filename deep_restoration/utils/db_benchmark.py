import os.path

import numpy as np
import skimage.io
import tensorflow as tf

from modules.inv_default_modules import get_stacked_module
from modules.loss_modules import NormedMSELoss, MSELoss, VggScoreLoss
from net_inversion import NetInversion
from utils.filehandling import load_image
from utils.rec_evaluation import subset10_paths, selected_img_ids, classifier_stats


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


def db_img_mse_and_vgg_scores(classifier, select_modules=None, select_images=None, merged=True):
    tgt_paths = subset10_paths(classifier)
    _, img_hw, layer_names = classifier_stats(classifier)
    tgt_images = [load_image(p) for p in tgt_paths]

    _, img_hw, layer_names = classifier_stats(classifier)
    log_path = '../logs/cnn_inversion/{}/'.format(classifier)

    stack_mode = 'merged' if merged else 'stacked'
    start_module_ids = select_modules or (1, 4, 7, 8, 9)
    img_subdirs = select_images or selected_img_ids()

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

            img_list = []
            for module_id in start_module_ids:
                layer_log_path = '{}stack_{}_to_1/{}/'.format(log_path, module_id, stack_mode)
                layer_list = []
                for idx, img_subdir in enumerate(img_subdirs):

                    rec_image = load_image(layer_log_path + 'img_rec_{}.png'.format(img_subdir))
                    if rec_image.shape[0] == 1:
                        rec_image = np.squeeze(rec_image, axis=0)

                    scores = sess.run(loss_tsr_list, feed_dict={tgt_pl: tgt_images[idx], rec_pl: rec_image})
                    layer_list.append(scores)

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
    for start_layer, rec_layer in lin_start_rec_pairs:

        run_stacked_module(classifier, start_layer, rec_layer, use_solotrain=use_solotrain,
                           subdir_name=None)


def db_collect_rec_images(classifier, select_modules=None, select_images=None, merged=False):
    # makes one [layers, imgs, h, w, c] mat for all rec images

    _, img_hw, layer_names = classifier_stats(classifier)
    log_path = '../logs/cnn_inversion/{}/'.format(classifier)

    stack_mode = 'merged' if merged else 'stacked'
    start_module_ids = select_modules or (1, 4, 7, 8, 9)
    img_subdirs = select_images or selected_img_ids()

    img_list = []
    for module_id in start_module_ids:
        layer_log_path = '{}stack_{}_to_1/{}/'.format(log_path, module_id, stack_mode)
        layer_list = []
        for idx, img_subdir in enumerate(img_subdirs):

            rec_image = load_image(layer_log_path + 'img_rec_{}.png'.format(img_subdir))
            if rec_image.shape[0] == 1:
                rec_image = np.squeeze(rec_image, axis=0)
            layer_list.append(rec_image)

        img_list.append(layer_list)

    return np.asarray(img_list)
