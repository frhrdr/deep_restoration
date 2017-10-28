import os.path

import numpy as np
import skimage.io
import tensorflow as tf

from modules.loss_modules import NormedMSELoss, SoftRangeLoss, TotalVariationLoss, VggScoreLoss, MSELoss
from modules.norm_module import NormModule
from modules.split_module import SplitModule
from net_inversion import NetInversion
from utils.db_benchmark import classifier_stats, subset10_paths
from utils.filehandling import load_image


def mv_script_fun(src_layer, img_path, log_path, classifier,
                  n_iterations=3500, lr_lower_points=None, jitter_stop_point=2750, range_b=80, alpha=6, beta=2):

    if not src_layer.endswith(':0'):
        src_layer = src_layer + ':0'

    imagenet_mean, img_hw, _ = classifier_stats(classifier)
    mse_weight, jitter_t = get_jitter_and_mse_weight(classifier, src_layer)

    sr_weight = 1. / (img_hw ** 2 * range_b ** alpha)
    tv_weight = 1. / (img_hw ** 2 * (range_b / 6.5) ** beta)

    split = SplitModule(name_to_split=src_layer, img_slice_name='img_rep', rec_slice_name='rec_rep')
    norm = NormModule(name_to_norm='pre_featmap/read:0', out_name='pre_featmap_normed', offset=imagenet_mean, scale=1.0)
    mse = NormedMSELoss(target='img_rep:0', reconstruction='rec_rep:0', weighting=mse_weight)
    sr_prior = SoftRangeLoss(tensor='pre_featmap_normed:0', alpha=6, weighting=sr_weight)
    tv_prior = TotalVariationLoss(tensor='pre_featmap_normed:0', beta=2, weighting=tv_weight)

    modules = [split, norm, mse, sr_prior, tv_prior]

    lr_factor = range_b ** 2 / alpha
    lr_lower_points = lr_lower_points or ((0, 1e-2 * lr_factor), (1000, 3e-3 * lr_factor), (2000, 1e-3 * lr_factor))

    ni = NetInversion(modules, log_path, classifier=classifier, print_freq=100, log_freq=1750)
    pre_img_init = np.expand_dims(load_image(img_path), axis=0).astype(np.float32)

    ni.train_pre_featmap(img_path, n_iterations=n_iterations, optim_name='adam',
                         jitter_t=jitter_t, jitter_stop_point=jitter_stop_point, range_clip=True, scale_pre_img=1.,
                         range_b=range_b,
                         lr_lower_points=lr_lower_points,
                         pre_featmap_init=pre_img_init, ckpt_offset=0, save_as_plot=False)


def run_mv_scripts(classifier):
    _, img_hw, layer_names = classifier_stats(classifier)
    img_paths = subset10_paths(classifier)
    img_paths = img_paths[3:]  # to continue previous run
    for img_path in img_paths:
        for layer_name in layer_names:
            layer_subdir = layer_name.replace('/', '_')
            img_subdir = img_path.split('/')[-1].split('.')[0]
            log_path = '../logs/mahendran_vedaldi/2016/{}/{}/{}/'.format(classifier, layer_subdir, img_subdir)
            mv_script_fun(layer_name, img_path, log_path, classifier)


def get_jitter_and_mse_weight(classifier, layer_name):
    _, _, layers = classifier_stats(classifier)
    if layer_name.endswith(':0'):
        layer_name = layer_name[:-len(':0')]
    idx = layers.index(layer_name)
    if classifier == 'alexnet':
        mse_weights = (300, 300, 300, 300,
                       300, 300, 300, 300,
                       300, 300, 100, 100, 20, 20, 1,
                       1, 1, 1, 1, 1, 1, 1)
        jitter_t = (1, 1, 1, 2,
                    2, 2, 2, 4,
                    4, 4, 4, 4, 4, 4, 8,
                    8, 8, 8, 8, 8, 8, 8)
    else:
        mse_weights = (300, 300, 300, 300, 300,
                       300, 300, 300, 300, 300,
                       300, 300, 300, 300, 300, 300, 300,
                       300, 300, 300, 300, 300, 300, 100,
                       100, 100, 20, 20, 20, 20, 1,
                       1, 1, 1, 1, 1, 1, 1)
        jitter_t = (0, 0, 0, 0, 0,
                    0, 0, 0, 0, 1,
                    1, 1, 1, 1, 1, 1, 2,
                    2, 2, 2, 2, 2, 2, 4,
                    4, 4, 4, 4, 4, 4, 8,
                    8, 8, 8, 8, 8, 8, 8)
    return mse_weights[idx], jitter_t[idx]


def mv_plot_mats(classifier):
    _, img_hw, layer_names = classifier_stats(classifier)
    img_paths = subset10_paths(classifier)
    img_paths = img_paths[3:]  # to continue previous run
    log_points = (1750, 3500)
    for img_path in img_paths:
        for layer_name in layer_names:
            layer_subdir = layer_name.replace('/', '_')
            img_subdir = img_path.split('/')[-1].split('.')[0]
            log_path = '../logs/mahendran_vedaldi/2016/{}/{}/{}/'.format(classifier, layer_subdir, img_subdir)
            if not os.path.exists(log_path + 'imgs/'):
                os.makedirs(log_path + 'imgs/')
            for log_point in log_points:
                img_mat = np.load('{}mats/rec_{}.npy'.format(log_path, log_point))
                skimage.io.imsave('{}imgs/rec_{}.png'.format(log_path, log_point), img_mat)


def mv_mse_and_vgg_scores(classifier):
    tgt_paths = subset10_paths(classifier)
    _, img_hw, layer_names = classifier_stats(classifier)
    log_path = '../logs/mahendran_vedaldi/2016/{}/'.format(classifier)
    layer_subdirs = [l.replace('/', '_') for l in layer_names]
    img_subdirs = [p.split('/')[-1].split('.')[0] for p in tgt_paths]

    tgt_images = [np.expand_dims(load_image(p), axis=0) for p in tgt_paths]
    rec_filename = 'imgs/rec_3500.png'

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


def mv_aggregate_rec_images(classifier, select_layers=None, select_images=None):
    # makes one [layers, imgs, h, w, c] mat for all rec3500 images
    tgt_paths = subset10_paths(classifier)
    _, img_hw, layer_names = classifier_stats(classifier)
    log_path = '../logs/mahendran_vedaldi/2016/{}/'.format(classifier)
    layer_subdirs = select_layers or [l.replace('/', '_') for l in layer_names]
    img_subdirs = select_images or [p.split('/')[-1].split('.')[0] for p in tgt_paths]

    tgt_images = [np.expand_dims(load_image(p), axis=0) for p in tgt_paths]
    rec_filename = 'imgs/rec_3500.png'

    for layer_subdir in layer_subdirs:
        layer_log_path = '{}{}/'.format(log_path, layer_subdir)

        for idx, img_subdir in enumerate(img_subdirs):
            img_log_path = '{}{}/'.format(layer_log_path, img_subdir)
            rec_image = np.expand_dims(load_image(img_log_path + rec_filename), axis=0)