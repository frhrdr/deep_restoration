import os.path

import numpy as np
import skimage.io

from modules.loss_modules import NormedMSELoss, SoftRangeLoss, TotalVariationLoss
from modules.norm_module import NormModule
from modules.split_module import SplitModule
from net_inversion import NetInversion
from utils.filehandling import load_image
from utils.rec_evaluation import subset10_paths, classifier_stats


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
                         pre_featmap_init=pre_img_init, ckpt_offset=0, save_as_plot=True)


def run_mv_scripts(classifier, custom_layers=None):
    _, img_hw, layer_names = classifier_stats(classifier)
    layer_names = custom_layers or layer_names
    img_paths = subset10_paths(classifier)
    # img_paths = img_paths[3:]  # to continue previous run
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


def mv_collect_rec_images(classifier, select_layers=None, select_images=None):
    # makes one [layers, imgs, h, w, c] mat for all rec3500 images
    tgt_paths = subset10_paths(classifier)
    _, img_hw, layer_names = classifier_stats(classifier)
    log_path = '../logs/mahendran_vedaldi/2016/{}/'.format(classifier)
    layer_subdirs = select_layers or [l.replace('/', '_') for l in layer_names]
    img_subdirs = select_images or [p.split('/')[-1].split('.')[0] for p in tgt_paths]

    # tgt_images = [np.expand_dims(load_image(p), axis=0) for p in tgt_paths]
    rec_filename = 'imgs/rec_3500.png'
    img_list = []
    for layer_subdir in layer_subdirs:
        layer_log_path = '{}{}/'.format(log_path, layer_subdir)
        layer_list = []
        for idx, img_subdir in enumerate(img_subdirs):
            img_log_path = '{}{}/'.format(layer_log_path, img_subdir)
            rec_image = load_image(img_log_path + rec_filename)
            if rec_image.shape[0] == 1:
                rec_image = np.squeeze(rec_image, axis=0)
            layer_list.append(rec_image)

        img_list.append(layer_list)

    return np.asarray(img_list)
