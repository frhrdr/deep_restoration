from utils.preprocessing import make_flattened_patch_data, add_flattened_validation_set, \
    make_channel_separate_patch_data, add_channelwise_validation_set
from utils.filehandling import resize_all_images
from modules.foe_full_prior import FoEFullPrior
from modules.foe_channelwise_prior import FoEChannelwisePrior
from modules.foe_separable_prior import FoESeparablePrior
# resize_all_images((227, 227), 'images_resized_227')

# make_flattened_patch_data(num_patches=100000, ph=8, pw=8, classifier='alexnet', map_name='conv2/lin:0',
#                           n_channels=256,
#                           n_feats_white=16384, whiten_mode='pca', batch_size=100,
#                           mean_mode='gc', sdev_mode='gc',
#                           raw_mat_load_path='')
#
# add_flattened_validation_set(num_patches=1000, ph=8, pw=8, classifier='alexnet', map_name='conv2/lin:0',
#                              n_channels=256, n_feats_white=16384, whiten_mode='pca', batch_size=100,
#                              mean_mode='gc', sdev_mode='gc')
#
make_channel_separate_patch_data(num_patches=100000, ph=8, pw=8, classifier='alexnet', map_name='conv1/lin:0',

                                 n_channels=96, n_feats_per_channel_white=8*8, whiten_mode='zca', batch_size=100,
                                 mean_mode='gc', sdev_mode='gc',
                                 raw_mat_load_path='')

add_channelwise_validation_set(num_patches=1000, ph=8, pw=8, classifier='alexnet', map_name='conv1/lin:0',
                               n_channels=96, n_feats_per_channel_white=64, whiten_mode='zca',
                               batch_size=100, mean_mode='gc', sdev_mode='gc')

prior = FoESeparablePrior(tensor_names='conv1/lin:0', weighting=1e-10, classifier='alexnet',
                          filter_dims=[8, 8], input_scaling=1.0, n_components=9000, n_channels=96,
                          dim_multiplier=300, share_weights=True, n_features_per_channel_white=64,
                          dist='logistic', mean_mode='gc', sdev_mode='gc', whiten_mode='zca')

prior.train_prior(batch_size=500, n_iterations=15000,
                  lr_lower_points=((0, 1e-0), (6000, 1e-1), (6500, 3e-2),
                                   (7000, 1e-2), (7500, 3e-3), (8000, 1e-3),
                                   (9000, 1e-4), (10000, 3e-4), (12000, 1e-5)),
                  grad_clip=100.0,
                  n_data_samples=100000,
                  log_freq=1000, summary_freq=10, print_freq=100, test_freq=100, n_val_samples=1000,
                  prev_ckpt=0,
                  optimizer_name='adam')

# prior = FoEFullPrior(tensor_names='conv1/lin:0', weighting=1e-10, classifier='alexnet',
#                      filter_dims=[8, 8], input_scaling=1.0, n_components=6000, n_channels=96,
#                      n_features_white=3000, dist='student', mean_mode='gc', sdev_mode='gc',
#                      load_name='FoEPrior')
# #


# prior.plot_filters_all_channels(range(7), prior.load_path + 'filter_vis/')
# prior.plot_channels_top_filters(range(7), prior.load_path + 'filter_vis/top/')
