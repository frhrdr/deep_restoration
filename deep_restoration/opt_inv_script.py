import os
from shutil import copyfile
import numpy as np
from modules.core_modules import LossModule
from modules.foe_full_prior import FoEFullPrior
from modules.foe_separable_prior import FoESeparablePrior
from modules.loss_modules import MSELoss
from modules.split_module import lin_split_and_mse
from net_inversion import NetInversion
from utils.default_priors import get_default_prior

split1, mse1 = lin_split_and_mse(1, add_loss=False, mse_name='MSE_Reconstruction')
split2, mse2 = lin_split_and_mse(2, add_loss=True)
split3, mse3 = lin_split_and_mse(3, add_loss=True)
split4, mse4 = lin_split_and_mse(4, add_loss=True)
split5, mse5 = lin_split_and_mse(5, add_loss=True)
split6, mse6 = lin_split_and_mse(6, add_loss=True)

pre_mse = MSELoss(target='target_featmap/read:0', reconstruction='pre_featmap/read:0', name='MSE_Reconstruction')
pre_mse.add_loss = False


# fullprior = get_default_prior('fullc1l6000', custom_weighting=1e-4)
# slimprior = get_default_prior('slimc5l5000', custom_weighting=1e-7)

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


img_prior2 = get_default_prior('full512logistic', custom_weighting=1e-2)
# img_prior3 = FoEFullPrior('pre_featmap:0', 3e-3, 'alexnet', [8, 8], 1.0, n_components=512, n_channels=3,
#                           n_features_white=64*3-1, dist='logistic', mean_mode='lf', sdev_mode='gc', whiten_mode='pca',
#                           load_name='ICAPrior', load_tensor_names='image')
# img_prior3 = FoEFullPrior('pre_featmap:0', 3e-3, 'alexnet', [8, 8], 1.0, n_components=1024, n_channels=3,
#                           n_features_white=64*3-1, dist='logistic', mean_mode='lf', sdev_mode='gc', whiten_mode='pca',
#                           load_tensor_names='image')

# log_path = '../logs/opt_inversion/alexnet/slim_vs_img/c2l_to_c1l/full_prior/1e-4/'
# log_path = '../logs/opt_inversion/alexnet/slim_vs_img/c4l_to_c3l/pre_image_8x8_dropout_prior/1e-8/'
img_prior1 = get_default_prior('full512', custom_weighting=3e-5)
modules = [split5, mse5, pre_mse, img_prior1]
log_path = '../logs/opt_inversion/alexnet/img_prior_comp/c5l_to_img/8x8_gc_gc_student/3e-5/jitter_bound_plots/'
cutoff = 'conv5/lin'
jitter_t = 16
# log_path = '../logs/opt_inversion/alexnet/img_prior_comp/c3l_to_img/pure_mse/'
ni = NetInversion(modules, log_path, classifier='alexnet', summary_freq=10, print_freq=50, log_freq=500)

if not os.path.exists(log_path):
    os.makedirs(log_path)
copyfile('./opt_inv_script.py', log_path + 'script.py')

target_image = '../data/selected/images_resized_227/red-fox.bmp'
# pre_featmap_name = 'conv1/lin'
pre_featmap_name = 'input'
do_plot = True

jitter_stop_point = 3200
lr = 3e-1

# pre_featmap_init = None
pre_featmap_init = np.load('../logs/opt_inversion/alexnet/img_prior_comp/c2l_to_img/pure_mse/mats/rec_10000.npy')
ni.train_pre_featmap(target_image, n_iterations=500, grad_clip=10000.,
                     lr_lower_points=((1e+0, lr),), jitter_t=jitter_t, range_clip=False, bound_plots=True,
                     optim_name='adam', save_as_plot=do_plot, jitter_stop_point=jitter_stop_point,
                     pre_featmap_init=pre_featmap_init, ckpt_offset=0,
                     pre_featmap_name=pre_featmap_name, classifier_cutoff=cutoff,
                     featmap_names_to_plot=(), max_n_featmaps_to_plot=10)

pre_featmap_init = np.load(ni.log_path + 'mats/rec_500.npy')
for mod in ni.modules:
    if isinstance(mod, LossModule):
        mod.reset()

ni.train_pre_featmap(target_image, n_iterations=9500, grad_clip=10000., optim_name='adam',
                     lr_lower_points=((1e+0, lr),), jitter_t=jitter_t, range_clip=False, bound_plots=True,
                     pre_featmap_init=pre_featmap_init, ckpt_offset=500, jitter_stop_point=jitter_stop_point,
                     pre_featmap_name=pre_featmap_name, classifier_cutoff=cutoff,
                     featmap_names_to_plot=(), max_n_featmaps_to_plot=10, save_as_plot=do_plot)

# pre_featmap_init = np.load(ni.log_path + 'mats/rec_1500.npy')
# for mod in ni.modules:
#     if isinstance(mod, LossModule):
#         mod.reset()
#
# ni.train_pre_featmap(target_image, n_iterations=8500, grad_clip=10000., optim_name='adam',
#                      lr_lower_points=((1e+0, lr),), jitter_t=jitter_t,
#                      pre_featmap_init=pre_featmap_init, ckpt_offset=1500, jitter_stop_point=jitter_stop_point,
#                      pre_featmap_name=pre_featmap_name, classifier_cutoff=cutoff,
#                      featmap_names_to_plot=(), max_n_featmaps_to_plot=10, save_as_plot=do_plot)

# pre_featmap_init = np.load(ni.log_path + 'mats/rec_5000.npy')
# for mod in ni.modules:
#     if isinstance(mod, LossModule):
#         mod.reset()
#
# ni.train_pre_featmap(target_image, n_iterations=8500, grad_clip=10000., optim_name='adam',
#                      lr_lower_points=((1e+0, 6e-1),), jitter_t=jitter_t,
#                      pre_featmap_init=pre_featmap_init, ckpt_offset=5000, jitter_stop_point=jitter_stop_point,
#                      pre_featmap_name=pre_featmap_name, classifier_cutoff=cutoff,
#                      featmap_names_to_plot=(), max_n_featmaps_to_plot=10, save_as_plot=do_plot)
