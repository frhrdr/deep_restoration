import os
from shutil import copyfile
import numpy as np
from modules.core_modules import LossModule
from modules.foe_full_prior import FoEFullPrior
from modules.foe_separable_prior import FoESeparablePrior
from modules.loss_modules import MSELoss, TotalVariationLoss
from modules.split_module import SplitModule
from net_inversion import NetInversion
from utils.db_mv_comparison import subset10_paths, selected_img_ids, classifier_stats
from utils.default_priors import get_default_prior
from utils.filehandling import load_image


def run_image_opt_inversions(classifier, prior_mode):
    weight = None
    _, img_hw, layer_names = classifier_stats(classifier)

    tgt_paths = subset10_paths(classifier)
    # tgt_images = [np.expand_dims(load_image(p), axis=0) for p in tgt_paths]
    layer_subdirs = [n.replace('/', '_') for n in layer_names]
    img_subdirs = ['val{}'.format(i) for i in selected_img_ids()]

    log_path = '../logs/opt_inversion/{}/image_rec/'.format(classifier)

    for idx, layer_subdir in enumerate(layer_subdirs):
        cutoff = layer_names[idx]

        for idy, img_subdir in enumerate(img_subdirs):
            # tgt_image = tgt_images[idy]
            target_image = tgt_paths[idy]
            exp_log_path = '{}{}/{}/'.format(log_path, layer_subdir, img_subdir)
            if not os.path.exists(log_path):
                os.makedirs(log_path)

            pre_featmap_name = 'input'
            do_plot = True
            summary_freq = 10
            print_freq = 100
            log_freq = 500
            grad_clip = 10000.
            lr_lower_points = ((1e+0, 3e-1),)

            split = SplitModule(name_to_split=cutoff + ':0', img_slice_name=layer_subdir + '_img',
                                rec_slice_name=layer_subdir + '_rec')
            feat_mse = MSELoss(target=layer_subdir + '_img:0', reconstruction=layer_subdir + '_rec:0',
                               name='MSE_' + layer_subdir)
            img_mse = MSELoss(target='target_featmap/read:0', reconstruction='pre_featmap/read:0',
                              name='MSE_Reconstruction')
            img_mse.add_loss = False
            prior = get_default_prior(prior_mode, custom_weighting=weight)

            modules = [split, feat_mse, img_mse, prior]

            pre_featmap_init = None
            ni = NetInversion(modules, exp_log_path, classifier=classifier, summary_freq=summary_freq,
                              print_freq=print_freq, log_freq=log_freq)
            ni.train_pre_featmap(target_image, n_iterations=500, optim_name='adam',
                                 lr_lower_points=lr_lower_points, grad_clip=grad_clip,
                                 pre_featmap_init=pre_featmap_init, ckpt_offset=0,
                                 pre_featmap_name=pre_featmap_name, classifier_cutoff=cutoff,
                                 featmap_names_to_plot=(), max_n_featmaps_to_plot=10, save_as_plot=do_plot)

            pre_featmap_init = np.load(ni.log_path + 'mats/rec_500.npy')
            for mod in ni.modules:
                if isinstance(mod, LossModule):
                    mod.reset()

            ni.train_pre_featmap(target_image, n_iterations=9500, optim_name='adam',
                                 lr_lower_points=lr_lower_points, grad_clip=grad_clip,
                                 pre_featmap_init=pre_featmap_init, ckpt_offset=500,
                                 pre_featmap_name=pre_featmap_name, classifier_cutoff=cutoff,
                                 featmap_names_to_plot=(), max_n_featmaps_to_plot=10, save_as_plot=do_plot)


def get_imagerec_jitter_and_prior_weight(classifier, layer_name):
    _, _, layers = classifier_stats(classifier)
    if layer_name.endswith(':0'):
        layer_name = layer_name[:-len(':0')]
    idx = layers.index(layer_name)
    if classifier == 'alexnet':
        prior_weights = (300, 300, 300, 300,
                         300, 300, 300, 300,
                         300, 300, 100, 100, 20, 20, 1,
                         1, 1, 1, 1, 1, 1, 1)
        jitter_t = (1, 1, 1, 2,
                    2, 2, 2, 4,
                    4, 4, 4, 4, 4, 4, 8,
                    8, 8, 8, 8, 8, 8, 8)
    else:
        prior_weights = (300, 300, 300, 300, 300,
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
    return prior_weights[idx], jitter_t[idx]