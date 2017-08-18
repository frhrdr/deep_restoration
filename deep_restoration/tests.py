from utils.preprocessing import make_channel_separate_patch_data, make_flattened_patch_data
# from utils.temp_utils import plot_feat_map_diffs
# import numpy as np
from modules.channel_ica_prior import ChannelICAPrior
from modules.foe_prior import FoEPrior
# make_channel_separate_patch_data(100000, 5, 5, 'alexnet', 'conv1/lin:0', 96,
#                                  save_dir='../data/patches/alexnet/conv1_lin_5x5_24feats_mean_lc_sdev_rescaled_'
#                                           + str(1/255) + '_channelwise/',
#                                  whiten_mode='pca', batch_size=100,
#                                  mean_mode='lc', sdev_mode=1/255,
#                                  raw_mat_load_path='../data/patches/alexnet/conv1_lin_5x5_24feats_mean_gc_sdev_gc_channelwise/raw_mat.npy')

# c1l_prior = ChannelICAPrior('conv1/lin:0', 1e-10, 'alexnet', [5, 5], input_scaling=1.0,
#                             n_components=150, n_channels=96,
#                             n_features_white=24,
#                             trainable=False, name='ChannelICAPrior', mean_mode='lc', sdev_mode=1/255)

c1l_prior = FoEPrior(tensor_names='conv1/lin:0',
                     weighting=1e-12, name='FoEPrior',
                     classifier='alexnet',
                     filter_dims=[5, 5], input_scaling=1.0, n_components=6000, n_channels=96,
                     n_features_white=1800, mean_mode='lc', sdev_mode='gc')

# c1l_prior.train_prior(batch_size=500, num_iterations=20000,
#                       lr_lower_points=((0, 1e-1), (1000, 3e-2), (3000, 1e-2),
#                                        (5000, 3e-3), (10000, 1e-3), (15000, 3e-4)),
#                       log_freq=5000, summary_freq=10, print_freq=100, prev_ckpt=0,
#                       optimizer_name='adam', plot_filters=False)

c1l_prior.plot_filters(range(3), c1l_prior.load_path + 'filter_vis/')

# 5x5 flat mean: global channel, sdev: global channel 2400f
# make_flattened_patch_data(num_patches=100000, ph=5, pw=5, classifier='alexnet', map_name='conv1/lin:0',
#                           n_channels=96,
#                           save_dir='../data/patches/alexnet/conv1_lin_5x5_1200feats_mean_lc_sdev_gc/',
#                           n_feats_white=1200, whiten_mode='pca', batch_size=100,
#                           mean_mode='lc', sdev_mode='gc',
#                           raw_mat_load_path='../data/patches/alexnet/new/conv1_lin_5x5_2399feats_mean_gc_sdev_gc/raw_mat.npy')
