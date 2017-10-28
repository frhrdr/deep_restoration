import os
from shutil import copyfile
import numpy as np
from modules.core_modules import LossModule
from modules.loss_modules import MSELoss
from modules.split_module import lin_split_and_mse
from net_inversion import NetInversion
from utils.default_priors import get_default_prior
from modules.split_module import SplitModule

pre_mse = MSELoss(target='target_featmap/read:0', reconstruction='pre_featmap/read:0', name='MSE_Reconstruction')
pre_mse.add_loss = False
# chanprior = FoEChannelwisePrior(tensor_names='pre_featmap/read:0', weighting=6e-5, classifier='alexnet',
#                                 filter_dims=[8, 8], input_scaling=1.0, n_components=150, n_channels=96,
#                                 n_features_per_channel_white=64,
#                                 dist='logistic', mean_mode='gc', sdev_mode='gc', whiten_mode='zca',
#                                 load_tensor_names='conv1/lin:0')


# tv_prior = TotalVariationLoss(tensor='pre_featmap/read:0', beta=2, weighting=1e-10)

# imgprior = FoEFullPrior('pre_featmap/read:0', 1e-2, 'alexnet', [12, 12], 1.0, n_components=1000, n_channels=3,
#                         n_features_white=12**2*3, dist='student', mean_mode='gc', sdev_mode='gc', whiten_mode='pca',
#                         name=None, load_name=None, dir_name=None, load_tensor_names='image')

# p = FoESeparablePrior('rgb_scaled:0', 1e-10, 'alexnet', [9, 9], 1.0, n_components=500, n_channels=3,
#                       n_features_per_channel_white=9**2,
#                       dim_multiplier=50, share_weights=False, channelwise_data=True,
#                       dist='student', mean_mode='gc', sdev_mode='gc', whiten_mode='zca',
#                       name=None, load_name=None, dir_name=None, load_tensor_names=None)
# dropout_prior = get_default_prior('dropout1024', custom_weighting=1e-8)
# img_prior2 = get_default_prior('full512logistic', custom_weighting=1e-2)
layer = 'softmax'
jitter_t = 8  # 1:1, 2:2, 3,4,5:4, 6,7,8:8
weighting = '1e-1'
make_mse = False
restart_adam = False

img_prior1 = get_default_prior('full512', custom_weighting=float(weighting))
if layer == 'softmax':
    split = SplitModule(name_to_split='softmax:0', img_slice_name='smx_img',
                        rec_slice_name='smx_rec', name='Split_smx')
    mse = MSELoss(target='smx_img' + ':0', reconstruction='smx_rec' + ':0',
                  name='MSE_smx')
    subdir = '8x8_gc_gc_student/{}/jitter_bound_plots/'.format(weighting)
    log_dir = '../logs/opt_inversion/alexnet/img_prior_comp/smx_to_img/'
    cutoff = None
else:
    subdir = '8x8_gc_gc_student/{}/jitter_bound_plots/'.format(weighting)
    log_dir = '../logs/opt_inversion/alexnet/img_prior_comp/c{}l_to_img/'.format(layer)
    cutoff = 'conv{}/lin'.format(layer) if layer < 6 else None
    split, mse = lin_split_and_mse(layer, add_loss=True)

if not make_mse:
    modules = [split, mse, pre_mse, img_prior1]
    log_path = log_dir + subdir
    pre_featmap_init = np.load(log_dir + 'pure_mse/mats/rec_10000.npy')
else:
    modules = [split, mse, pre_mse]
    log_path = log_dir + 'pure_mse/'
    pre_featmap_init = None
ni = NetInversion(modules, log_path, classifier='alexnet', summary_freq=10, print_freq=50, log_freq=500)

if not os.path.exists(log_path):
    os.makedirs(log_path)
copyfile('./opt_inv_script.py', log_path + 'script.py')

target_image = '../data/selected/images_resized_227/red-fox.bmp'
pre_featmap_name = 'input'
do_plot = True

jitter_stop_point = 3200
lr = 1.
bound_plots = True

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
                         lr_lower_points=((1e+0, lr),), jitter_t=jitter_t, range_clip=False, bound_plots=bound_plots,
                         pre_featmap_init=pre_featmap_init, ckpt_offset=500, jitter_stop_point=jitter_stop_point,
                         pre_featmap_name=pre_featmap_name, classifier_cutoff=cutoff,
                         featmap_names_to_plot=(), max_n_featmaps_to_plot=10, save_as_plot=do_plot)
