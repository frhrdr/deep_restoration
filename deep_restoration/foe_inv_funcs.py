import os
from shutil import copyfile
import numpy as np
import tensorflow as tf
from modules.core_modules import LossModule
from modules.loss_modules import MSELoss, VggScoreLoss, NormedMSELoss
from modules.split_module import SplitModule, lin_split_and_mse
from net_inversion import NetInversion
from utils.rec_evaluation import subset10_paths, selected_img_ids, classifier_stats
from utils.default_priors import get_default_prior
from utils.filehandling import load_image
from skimage.io import imsave


def run_image_opt_inversions(classifier, prior_mode, layer_select=None, lr=1., pre_featmap_name='input',
                             do_plot=True, mse_iterations=5000, opt_iterations=5000, jitterations=3200,
                             summary_freq=50, print_freq=500, log_freq=500, grad_clip=10000., select_img=None):

    _, img_hw, layer_names = classifier_stats(classifier)

    if layer_select is not None:
        layer_names = [n for n in layer_names if layer_select in n]

    tgt_paths = subset10_paths(classifier)
    layer_subdirs = [n.replace('/', '_') for n in layer_names]
    img_subdirs = ['val{}'.format(i) for i in selected_img_ids()]

    if select_img is not None:
        tgt_paths = tgt_paths[select_img:select_img+1]
        img_subdirs = img_subdirs[select_img:select_img+1]
    log_path = '../logs/opt_inversion/{}/image_rec/'.format(classifier)
    print(layer_names)
    for idx, layer_subdir in enumerate(layer_subdirs):
        cutoff = None  # layer_names[idx] if layer_names[idx].startswith('conv') else None
        jitter_t, weight = get_imagerec_jitter_and_prior_weight(classifier, layer_names[idx])
        print('jitter', jitter_t, 'prior_weight', weight)
        for idy, img_subdir in enumerate(img_subdirs):
            target_image = tgt_paths[idy]
            exp_log_path = '{}{}/{}/'.format(log_path, layer_subdir, img_subdir)
            if not os.path.exists(log_path):
                os.makedirs(log_path)

            lr_lower_points = ((1e+0, lr),)
            print(layer_subdir)
            split = SplitModule(name_to_split=layer_names[idx] + ':0', img_slice_name=layer_subdir + '_img',
                                rec_slice_name=layer_subdir + '_rec')
            feat_mse = MSELoss(target=layer_subdir + '_img:0', reconstruction=layer_subdir + '_rec:0',
                               name='MSE_' + layer_subdir)
            img_mse = MSELoss(target='target_featmap/read:0', reconstruction='pre_featmap/read:0',
                              name='MSE_Reconstruction')
            img_mse.add_loss = False

            modules = [split, feat_mse, img_mse]
            pure_mse_path = exp_log_path + 'pure_mse/'
            ni = NetInversion(modules, pure_mse_path, classifier=classifier, summary_freq=summary_freq,
                              print_freq=print_freq, log_freq=log_freq)

            pre_featmap_init = None
            ni.train_pre_featmap(target_image, n_iterations=mse_iterations, grad_clip=grad_clip,
                                 lr_lower_points=lr_lower_points, jitter_t=jitter_t, range_clip=False,
                                 bound_plots=True,
                                 optim_name='adam', save_as_plot=do_plot, jitter_stop_point=jitterations,
                                 pre_featmap_init=pre_featmap_init, ckpt_offset=0,
                                 pre_featmap_name=pre_featmap_name, classifier_cutoff=cutoff,
                                 featmap_names_to_plot=(), max_n_featmaps_to_plot=10)

            for mod in modules:
                if isinstance(mod, LossModule):
                    mod.reset()

            prior = get_default_prior(prior_mode, custom_weighting=weight, custom_target='pre_featmap:0')
            modules = [split, feat_mse, img_mse, prior]
            prior_path = exp_log_path + prior_mode + '/'
            ni = NetInversion(modules, prior_path, classifier=classifier, summary_freq=summary_freq,
                              print_freq=print_freq, log_freq=log_freq)
            pre_featmap_init = np.load(pure_mse_path + '/mats/rec_{}.npy'.format(mse_iterations))
            ni.train_pre_featmap(target_image, n_iterations=opt_iterations, grad_clip=grad_clip,
                                 lr_lower_points=lr_lower_points, jitter_t=jitter_t, range_clip=False,
                                 bound_plots=True,
                                 optim_name='adam', save_as_plot=do_plot,
                                 jitter_stop_point=mse_iterations + jitterations,
                                 pre_featmap_init=pre_featmap_init, ckpt_offset=mse_iterations,
                                 pre_featmap_name=pre_featmap_name, classifier_cutoff=cutoff,
                                 featmap_names_to_plot=(), max_n_featmaps_to_plot=10)


def get_imagerec_jitter_and_prior_weight(classifier, layer_name):
    _, _, layers = classifier_stats(classifier)
    if layer_name.endswith(':0'):
        layer_name = layer_name[:-len(':0')]
    idx = layers.index(layer_name)
    if classifier == 'alexnet':
        prior_weights = (1e-6, 1e-4, 3e-4, 1e-3,
                         3e-3, 3e-3, 3e-3, 1e-3,
                         1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4,
                         1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4)
        jitter_t = (0, 0, 1, 2,
                    2, 2, 2, 4,
                    4, 4, 4, 4, 4, 4, 8,
                    8, 8, 8, 8, 8, 8, 8)
    else:
        prior_weights = (1e-4, 1e-4, 1e-4, 1e-4, 1e-3,
                         1e-4, 1e-4, 1e-4, 1e-4, 1e-3,
                         1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-3,
                         1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-3,
                         1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-3,
                         1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4)
        jitter_t = (0, 0, 0, 0, 0,
                    0, 0, 0, 0, 1,
                    1, 1, 1, 1, 1, 1, 2,
                    2, 2, 2, 2, 2, 2, 4,
                    4, 4, 4, 4, 4, 4, 8,
                    8, 8, 8, 8, 8, 8, 8)
    return jitter_t[idx], prior_weights[idx]


def inv_mse_and_vgg_scores(classifier):
    tgt_paths = subset10_paths(classifier)
    _, img_hw, layer_names = classifier_stats(classifier)
    log_path = '../logs/opt_inversion/{}/image_rec/'.format(classifier)
    layer_subdirs = [n.replace('/', '_') for n in layer_names]
    img_subdirs = ['val{}'.format(i) for i in selected_img_ids()]

    tgt_images = [np.expand_dims(load_image(p), axis=0) for p in tgt_paths]
    rec_filename = 'imgs/rec_5000.png'

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
                layer_log_path = '{}{}/imgs/'.format(log_path, layer_subdir)
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


def foe_collect_rec_images(classifier, select_layers=None, select_images=None, rescaled=False):
    # makes one [layers, imgs, h, w, c] mat for all rec3500 images
    tgt_paths = subset10_paths(classifier)
    _, img_hw, layer_names = classifier_stats(classifier)
    log_path = '../logs/opt_inversion/{}/image_rec/'.format(classifier)
    layer_subdirs = select_layers or [l.replace('/', '_') for l in layer_names]
    img_subdirs = select_images or [p.split('/')[-1].split('.')[0] for p in tgt_paths]

    # tgt_images = [np.expand_dims(load_image(p), axis=0) for p in tgt_paths]
    scale_subdir = 'imgs_rescaled' if rescaled else 'imgs'
    rec_filename = 'full512/{}/rec_10000.png'.format(scale_subdir)
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


def foe_replot_rescaled(classifier='alexnet'):
    _, img_hw, layer_names = classifier_stats(classifier)
    log_path = '../logs/opt_inversion/{}/image_rec/'.format(classifier)
    layer_subdirs = [l.replace('/', '_') for l in layer_names if 'lin' in l]
    img_subdirs = ['val{}'.format(i) for i in selected_img_ids()]

    load_filename = 'full512/mats/rec_10000.npy'
    save_filename = 'full512/imgs_rescaled/rec_10000.png'

    for layer_subdir in layer_subdirs:
        layer_log_path = '{}{}/'.format(log_path, layer_subdir)
        for idx, img_subdir in enumerate(img_subdirs):
            img_log_path = '{}{}/'.format(layer_log_path, img_subdir)
            if not os.path.exists(img_log_path + 'full512/imgs_rescaled/'):
                os.makedirs(img_log_path + 'full512/imgs_rescaled/')
            mat = np.squeeze(np.load(img_log_path + load_filename), axis=0)
            mat = (mat - np.min(mat)) / (np.max(mat) - np.min(mat))
            imsave(img_log_path + save_filename, mat)


def run_featmap_opt_inversions():
    pass


def featmap_inv(match_layer, target_layer, image_name, prior_id, prior_weighting, make_mse=False, restart_adam=False,
                pre_image=True, do_plot=True, jitter_t=0, jitter_stop_point=3200, lr=1., bound_plots=True,
                custom_target=None):

    do_plot = do_plot if pre_image else False
    pre_img_subdir = 'pre_img' if pre_image else 'featmap'
    prior_subdir = 'no_prior' if make_mse else prior_id

    subdir = '{}/{}/'.format(prior_subdir, prior_weighting)
    log_dir = '../logs/opt_inversion/alexnet/featmap_rec/{}l_to_{}l/{}/'.format(match_layer, target_layer,
                                                                                pre_img_subdir)
    cutoff = 'conv{}/lin'.format(match_layer) if match_layer < 6 else None
    split_match, mse_match = lin_split_and_mse(match_layer, add_loss=True, mse_name='')

    modules = [split_match, mse_match]

    if pre_image:
        pre_featmap_name = 'input'
        split_target, mse_target = lin_split_and_mse(target_layer, add_loss=False,
                                                     mse_name='MSE_{}l_reconstruction'.format(target_layer))
        pre_mse = MSELoss(target='target_featmap/read:0', reconstruction='pre_featmap/read:0', name='MSE_pre_image')
        pre_mse.add_loss = False
        modules.extend([split_target, mse_target, pre_mse])

        prior_target = custom_target or split_target.rec_slice_name + ':0'
        prior = get_default_prior(prior_id, custom_weighting=float(prior_weighting),
                                  custom_target=prior_target)
    else:
        pre_featmap_name = 'conv{}/lin'.format(target_layer) if target_layer < 6 \
            else 'fc{}/lin'.format(target_layer)
        pre_mse = MSELoss(target='target_featmap/read:0', reconstruction='pre_featmap/read:0',
                          name='MSE_{}l_reconstruction'.format(target_layer))
        pre_mse.add_loss = False
        modules.extend([pre_mse])

        prior_target = custom_target or 'pre_featmap/read:0'
        prior = get_default_prior(prior_id, custom_weighting=float(prior_weighting), custom_target=prior_target)

    if not make_mse:
        modules.append(prior)
        log_path = log_dir + subdir
        pre_featmap_init = np.load(log_dir + 'pure_mse/mats/rec_10000.npy')
    else:
        log_path = log_dir + 'pure_mse/'
        pre_featmap_init = None
    ni = NetInversion(modules, log_path, classifier='alexnet', summary_freq=10, print_freq=50, log_freq=500)

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    copyfile('./foe_inv_featmap_script.py', log_path + 'script.py')

    target_image = '../data/selected/images_resized_227/{}.bmp'.format(image_name)

    first_n_iterations = 500 if restart_adam else 10000

    ni.train_pre_featmap(target_image, n_iterations=first_n_iterations, grad_clip=10000.,
                         lr_lower_points=((1e+0, lr),), jitter_t=jitter_t, range_clip=False, bound_plots=bound_plots,
                         optim_name='adam', save_as_plot=do_plot, jitter_stop_point=jitter_stop_point,
                         pre_featmap_init=pre_featmap_init, ckpt_offset=0,
                         pre_featmap_name=pre_featmap_name, classifier_cutoff=cutoff,
                         featmap_names_to_plot=(), max_n_featmaps_to_plot=10)

    if restart_adam:
        pre_featmap_init = np.load(ni.log_path + 'mats/rec_500.npy')
        for mod in ni.modules:
            if isinstance(mod, LossModule):
                mod.reset()

        ni.train_pre_featmap(target_image, n_iterations=9500, grad_clip=10000., optim_name='adam',
                             lr_lower_points=((1e+0, lr),), jitter_t=jitter_t, range_clip=False,
                             bound_plots=bound_plots,
                             pre_featmap_init=pre_featmap_init, ckpt_offset=500, jitter_stop_point=jitter_stop_point,
                             pre_featmap_name=pre_featmap_name, classifier_cutoff=cutoff,
                             featmap_names_to_plot=(), max_n_featmaps_to_plot=10, save_as_plot=do_plot)
