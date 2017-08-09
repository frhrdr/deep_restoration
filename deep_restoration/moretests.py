from utils.preprocessing import make_channel_separate_patch_data, make_flattened_patch_data
from modules.channel_ica_prior import ChannelICAPrior
from modules.ica_prior import ICAPrior
from modules.foe_prior import FoEPrior
from utils.temp_utils import patch_data_vis
# make_channel_separate_patch_data(num_patches=100000, ph=8, pw=8, classifier='alexnet', map_name='rgb_scaled:0',
#                                  n_channels=3,
#                                  save_dir='../data/patches/alexnet/8x8_mean_gc_sdev_gc_channelwise/',
#                                  whiten_mode='pca', batch_size=100,
#                                  mean_mode='gc', sdev_mode='global_channel')

# make_flattened_patch_data(num_patches=100000, ph=8, pw=8, classifier='alexnet', map_name='rgb_scaled:0',
#                           n_channels=3,
#                           save_dir='../data/patches/image/new/8x8_mean_gc_sdev_gc/',
#                           n_feats_white=191, whiten_mode='pca', batch_size=100,
#                           mean_mode='global_channel', cov_mode='global_channel')

# r = range(500, 17000, 500)
# for i in r:
#     mat_to_img('../logs/net_inversion/alexnet/c4_rec/mse4_pimg_pc2rich_1e-9/mats/rec_' + str(i) + '.npy', cols=1)
# mat_to_img('../logs/net_inversion/alexnet/c3_rec/mse3_foe2lin_1e-8/mats/rec_10000.npy', cols=1)

# make_channel_separate_feat_map_mats(num_patches=100000, ph=8, pw=8, classifier='alexnet',
#                                     map_name='conv1/lin:0', n_channels=96,
#                                     save_dir='../data/patches/alexnet/conv1_lin_8x8_63feats_channelwise/',
#                                     whiten_mode='pca', batch_size=100)
#
# make_feat_map_mats(100000, map_name='conv1/lin:0', classifier='alexnet', ph=5, pw=5,
#                    save_dir='../data/patches/alexnet/conv1_lin_5x5_2399feats/', whiten_mode='pca', batch_size=10)


#
# cica_prior = ChannelICAPrior(tensor_names='pre_img:0',
#                              weighting=1e-9, name='ChannelICAPrior',
#                              classifier='alexnet',
#                              filter_dims=[8, 8], input_scaling=1.0, n_components=150, n_channels=3,
#                              n_features_white=63, mean_mode='gc', sdev_mode='gc')
#
# cica_prior.train_prior(batch_size=500, num_iterations=10000, lr=3e-5,
#                        lr_lower_points=((0, 1e-3), (5000, 1e-4)),
#                        grad_clip=100.0, n_vis=144,
#                        whiten_mode='pca', num_data_samples=100000,
#                        log_freq=5000, summary_freq=10, print_freq=100, prev_ckpt=0, optimizer_name='adam',
#                        plot_filters=True, do_clip=True)

ica_prior = ICAPrior(tensor_names='pre_img:0',
                     weighting=1e-9, name='ICAPrior',
                     classifier='alexnet',
                     filter_dims=[8, 8], input_scaling=1.0, n_components=512, n_channels=3,
                     n_features_white=191, mean_mode='gc', sdev_mode='gc')

ica_prior.train_prior(batch_size=500, num_iterations=10000, lr=3e-5,
                      lr_lower_points=((0, 1e-1), (10000, 3e-2),),
                      grad_clip=100.0, n_vis=144,
                      whiten_mode='pca', num_data_samples=100000,
                      log_freq=5000, summary_freq=10, print_freq=100, prev_ckpt=10000, optimizer_name='adam',
                      plot_filters=True, do_clip=True)

# patch_data_vis('../data/patches/image/8x8_mean_gc_sdev_gc_channelwise/normed_mat.npy',
#                mat_shape=(100000, 3, 64), patch_hw=8)
#
# patch_data_vis('../data/patches/image/8x8_channelwise/data_mat_raw_channel.npy',
#                mat_shape=(100000, 3, 64), patch_hw=8)