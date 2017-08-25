from utils.preprocessing import make_channel_separate_patch_data, make_flattened_patch_data
# from utils.temp_utils import plot_feat_map_diffs
# import numpy as np
from modules.channel_ica_prior import ChannelICAPrior
from modules.foe_prior import FoEPrior
from utils.temp_utils import plot_alexnet_filters

# plot_alexnet_filters('./', filter_name='conv1', filter_ids=range(10))

# make_channel_separate_patch_data(100000, 5, 5, 'alexnet', 'conv2/lin:0', 256,
#                                  save_dir='../data/patches/alexnet/conv2_lin_5x5_24feats_mean_gc_sdev_gc_channelwise/',
#                                  whiten_mode='pca', batch_size=100,
#                                  mean_mode='gc', sdev_mode='gc',
#                                  raw_mat_load_path='')

# c2l_prior = ChannelICAPrior('conv2/lin:0', 1e-10, 'alexnet', [5, 5], input_scaling=1.0,
#                             n_components=150, n_channels=256,
#                             n_features_white=24,
#                             mean_mode='gc', sdev_mode='gc')

# c2l_prior.train_prior(batch_size=500, num_iterations=30000, lr=3e-5,
#                       lr_lower_points=(  # (0, 1e-0), (1000, 1e-1), (3000, 3e-2),
#                                        (5000, 1e-2), (6000, 3e-3), (7000, 1e-3),
#                                        (8000, 1e-4), (9000, 3e-5), (10000, 1e-5),),
#                       grad_clip=100.0,
#                       whiten_mode='pca', num_data_samples=100000,
#                       log_freq=1000, summary_freq=10, print_freq=100,
#                       prev_ckpt=8000,
#                       optimizer_name='adam')

# c1l_prior = FoEPrior(tensor_names='pre_img:0',
#                      weighting=1e-12, name='FoEPrior',
#                      classifier='alexnet',
#                      filter_dims=[8, 8], input_scaling=1.0, n_components=4000, n_channels=3,
#                      n_features_white=64*3-1, mean_mode='gc', sdev_mode='gc')
#
# c1l_prior.train_prior(batch_size=100, num_iterations=20000,
#                       lr_lower_points=((0, 1e-0), (4000, 1e-1), (6000, 3e-2), (8000, 1e-2),
#                                        (10000, 3e-3), (13000, 1e-3), (15000, 3e-4)),
#                       log_freq=1000, summary_freq=10, print_freq=100, prev_ckpt=6000,
#                       optimizer_name='adam', plot_filters=True)


# c1l_prior = ChannelICAPrior('conv1_lin:0', 1e-6, 'alexnet', [5, 5], input_scaling=1.0,
#                             n_components=150, n_channels=96,
#                             n_features_white=24,
#                             trainable=False, name='ChannelICAPrior', mean_mode='gc', sdev_mode='gc')

# c1l_prior = FoEPrior(tensor_names='conv1/lin:0',
#                      weighting=1e-12, name='FoEPrior',
#                      classifier='alexnet',
#                      filter_dims=[5, 5], input_scaling=1.0, n_components=6000, n_channels=96,
#                      n_features_white=1800, mean_mode='gc', sdev_mode='gc')

c1l_prior = FoEPrior(tensor_names='pool1:0',
                     weighting=1e-12, name='FoEPrior',
                     classifier='alexnet',
                     filter_dims=[5, 5], input_scaling=1.0, n_components=4800, n_channels=96,
                     n_features_white=2400, mean_mode='gc', sdev_mode='gc')

c1l_prior.plot_filters_all_channels(range(5), c1l_prior.load_path + 'filter_vis/')
c1l_prior.plot_channels_top_filters(range(5), c1l_prior.load_path + 'filter_vis/top/')


# 5x5 flat mean: global channel, sdev: global channel 2400f
# make_flattened_patch_data(num_patches=100000, ph=5, pw=5, classifier='alexnet', map_name='conv1/lin:0',
#                           n_channels=96,
#                           save_dir='../data/patches/alexnet/conv1_lin_5x5_1200feats_mean_lc_sdev_gc/',
#                           n_feats_white=1200, whiten_mode='pca', batch_size=100,
#                           mean_mode='lc', sdev_mode='gc',
#                           raw_mat_load_path='../data/patches/alexnet/new/conv1_lin_5x5_2399feats_mean_gc_sdev_gc/raw_mat.npy')
