import numpy as np
import tensorflow as tf
import skimage.io
import os.path
from net_inversion import NetInversion
from modules.inv_default_modules import get_stacked_module
from modules.split_module import SplitModule
from modules.loss_modules import NormedMSELoss, SoftRangeLoss, TotalVariationLoss, MSELoss, VggScoreLoss
from modules.norm_module import NormModule
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
                    scores = sess.run(loss_tsr_list, feed_dict={tgt_pl: tgt_images[idx], rec_pl: rec_image})
                    layer_score_list.append(scores)
                score_list.append(layer_score_list)

    score_mat = np.asarray(score_list)
    print(score_mat.shape)
    print(found_layers)
    np.save('{}score_mat.npy'.format(log_path), score_mat)


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
    lin_tensor_names = ['DC{}/c{}l_rec:0'.format(i, lin_start_rec_pairs[i][1]) for i in range(1, 5)]
    for lin_tensor, pair in zip(lin_start_rec_pairs, lin_tensor_names):
        start_layer, rec_layer = pair
        run_stacked_module(classifier, start_layer, rec_layer, use_solotrain=use_solotrain,
                           subdir_name=None, retrieve_special=lin_tensor)


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
