from utils.preprocessing import make_channel_separate_patch_data, make_flattened_patch_data
from modules.channel_ica_prior import ChannelICAPrior
from modules.ica_prior import ICAPrior
from modules.foe_prior import FoEPrior
from utils.temp_utils import patch_data_vis
import numpy as np
# make_channel_separate_patch_data(num_patches=100000, ph=8, pw=8, classifier='alexnet', map_name='rgb_scaled:0',
#                                  n_channels=3,
#                                  save_dir='../data/patches/alexnet/8x8_mean_gc_sdev_gc_channelwise/',
#                                  whiten_mode='pca', batch_size=100,
#                                  mean_mode='gc', sdev_mode='global_channel')
#
# make_flattened_patch_data(num_patches=100000, ph=5, pw=5, classifier='alexnet', map_name='rgb_scaled:0',
#                           n_channels=96,
#                           save_dir='../data/patches/alexnet/conv1_lin_5x5_1200feats_mean_lf_sdev_rescaled_' + str(1/255) + '/',
#                           n_feats_white=1200, whiten_mode='pca', batch_size=100,
#                           mean_mode='lf', sdev_mode=1/255,
#                           raw_mat_load_path='../data/patches/alexnet/conv1_lin_1200feats_mean_gc_sdev_gc/raw_mat.npy')
# 0.00392156862745098
# path ='../data/patches/alexnet/lrn1_5x5_1000feats_mean_lc_sdev_gc/normed_mat.npy'
# path = '../data/patches/alexnet/conv1_lin_5x5_1800feats_mean_lc_sdev_gc/normed_mat.npy'
#
# data_mat = np.memmap(path, dtype=np.float32, mode='r',
#                      shape=(100000, 96, 25))

ica_prior = ICAPrior(tensor_names='lrn1:0',
                     weighting=1e-9, name='ICAPrior',
                     classifier='alexnet',
                     filter_dims=[5, 5], input_scaling=1.0, n_components=2000, n_channels=96,
                     n_features_white=1000, mean_mode='lc', sdev_mode='gc')

ica_prior.train_prior(batch_size=500, num_iterations=10000, lr=3e-5,
                      lr_lower_points=((0, 1e-0), (3000, 1e-1), (5000, 1e-2), (10000, 3e-3), (15000, 1e-3),),
                      grad_clip=100.0, n_vis=144,
                      whiten_mode='pca', num_data_samples=100000,
                      log_freq=1000, summary_freq=10, print_freq=100,
                      prev_ckpt=000, optimizer_name='adam',
                      plot_filters=False, do_clip=True)

# ica_prior.plot_filters(range(1), ica_prior.load_path + 'filter_vis/')

# cica_prior = ChannelICAPrior(tensor_names='pre_img:0',
#                              weighting=1e-9, name='ChannelICAPrior',
#                              classifier='alexnet',
#                              filter_dims=[8, 8], input_scaling=1.0, n_components=150, n_channels=3,
#                              n_features_white=63, mean_mode='gc', sdev_mode='gc')

# cica_prior.train_prior(batch_size=500, num_iterations=10000, lr=3e-5,
#                        lr_lower_points=((0, 1e-3), (5000, 1e-4)),
#                        grad_clip=100.0, n_vis=144,
#                        whiten_mode='pca', num_data_samples=100000,
#                        log_freq=5000, summary_freq=10, print_freq=100, prev_ckpt=0, optimizer_name='adam',
#                        plot_filters=True, do_clip=True)

# foe_prior = FoEPrior(tensor_names='pre_img:0',
#                      weighting=1e-9, name='FoEPrior',
#                      classifier='alexnet',
#                      filter_dims=[8, 8], input_scaling=1.0, n_components=512, n_channels=3,
#                      n_features_white=191, mean_mode='lf', sdev_mode='none')

# foe_prior.train_prior(batch_size=500, num_iterations=30000, lr=3e-5,
#                       lr_lower_points=((0, 1e-1), (10000, 3e-2), (20000, 1e-2), (25000, 1e-3), (30000, 1e-4)),
#                       grad_clip=100.0, n_vis=144,
#                       whiten_mode='pca', num_data_samples=100000,
#                       log_freq=5000, summary_freq=10, print_freq=100, prev_ckpt=30000, optimizer_name='adam',
#                       plot_filters=True, do_clip=True)


# patch_data_vis('../data/patches/image/8x8_mean_gc_sdev_gc/normed_mat.npy',
#                mat_shape=(100000, 3, 64), patch_hw=8)

# patch_data_vis('../data/patches/image/8x8_mean_gc_sdev_gc/normed_mat.npy',
#                mat_shape=(100000, 3, 64), patch_hw=8)