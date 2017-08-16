# from utils.preprocessing import make_channel_separate_patch_data, make_flattened_patch_data
# from utils.temp_utils import plot_feat_map_diffs
# import numpy as np
from modules.ica_prior import ICAPrior

c1l_prior = ICAPrior(tensor_names='conv1_lin:0',
                     weighting=1e-14, name='C1LPrior',
                     classifier='alexnet',
                     filter_dims=[5, 5], input_scaling=1.0, n_components=3000, n_channels=96,
                     n_features_white=1800, mean_mode='lc', sdev_mode='gc')

c1l_prior.plot_filters(range(3), c1l_prior.load_path)
# 5x5 flat mean: global channel, sdev: global channel 2400f
# make_flattened_patch_data(num_patches=100000, ph=5, pw=5, classifier='alexnet', map_name='conv1/lin:0',
#                           n_channels=96,
#                           save_dir='../data/patches/alexnet/conv1_lin_5x5_1200feats_mean_lc_sdev_gc/',
#                           n_feats_white=1200, whiten_mode='pca', batch_size=100,
#                           mean_mode='lc', sdev_mode='gc',
#                           raw_mat_load_path='../data/patches/alexnet/new/conv1_lin_5x5_2399feats_mean_gc_sdev_gc/raw_mat.npy')
