from utils.preprocessing import make_channel_separate_patch_data, make_flattened_patch_data, \
    add_flattened_validation_set, add_channelwise_validation_set
from utils.temp_utils import show_patches_by_channel, plot_alexnet_filters
from modules.foe_separable_prior import FoESeparablePrior
from modules.foe_full_prior import FoEFullPrior
from modules.foe_channelwise_prior import FoEChannelwisePrior
import numpy as np
# from modules.foe_channelwise_prior import FoEChannelwisePrior
# from modules.foe_full_prior import FoEFullPrior
# from utils.temp_utils import plot_alexnet_filters, show_patches_by_channel
from modules.loss_modules import VggScoreLoss

# assert (1, 2, 3) == (1, 2, 3)
#
# v = VggScoreLoss(('target:0', 'reconstruction:0'), 1.0)
#
#
# tgt_path = '/home/frederik/PycharmProjects/deep_restoration/data/selected/images_resized/red-fox.bmp'
# rec_path = '/home/frederik/PycharmProjects/deep_restoration/logs/opt_inversion/alexnet/img/mats/rec_{}.npy'
# scores = []
#
# for idx in range(500, 13501, 500):
#     score = v.get_score(tgt_path, rec_path.format(idx), load_tgt_as_image=True)
#     print(score)
#     scores.append((idx,score))
# print(scores)

# p = FoEChannelwisePrior('rgb_scaled:0', 1e-10, 'alexnet', [8, 8], 1.0, n_components=200, n_channels=3,
#                         n_features_white=8**2-1, dist='student', mean_mode='gc', sdev_mode='gc',
#                         trainable=False, name=None, load_name=None, dir_name=None, load_tensor_names=None)
#
# p.train_prior(batch_size=500, n_iterations=15000, lr=3e-5,
#               lr_lower_points=((0, 1e-0), (1000, 1e-1),
#                                (2000, 3e-2),
#                                (3000, 1e-2), (4000, 3e-3), (5000, 1e-3),
#                                (8000, 1e-4), (10000, 3e-5), (13000, 1e-5),),
#               grad_clip=1e+100,
#               whiten_mode='pca', n_data_samples=100000, n_val_samples=1000,
#               log_freq=1000, summary_freq=10, print_freq=100,
#               prev_ckpt=0,
#               optimizer_name='adam', plot_filters=True, do_clip=True)

# img_prior = FoEFullPrior(tensor_names='rgb_scaled/read:0',
#                          weighting=1e-7, name='ICAPrior',
#                          classifier='alexnet',
#                          filter_dims=[12, 12], input_scaling=1.0, n_components=1000, n_channels=3,
#                          n_features_white=432,
#                          mean_mode='gc', sdev_mode='gc',
#                          load_tensor_names = 'image')
#
# img_prior.train_prior(10, 0, plot_filters=True, prev_ckpt=13000)
# hw = 9
# wmode = 'zca'
# nchan = 3

fullprior = FoEFullPrior('rgb_scaled:0', 1e-5, 'alexnet', [8, 8], 1.0, n_components=512, n_channels=3,
                         n_features_white=8 ** 2 * 3 - 1, dist='logistic', mean_mode='gc', sdev_mode='gc',
                         whiten_mode='pca')

fullprior.train_prior(batch_size=250, n_iterations=25000, lr=3e-5,
                      lr_lower_points=((0, 1e-0), (10000, 1e-1),
                                       (13000, 3e-2),
                                       (15000, 1e-2), (17000, 3e-3), (19000, 1e-3),
                                       (21000, 1e-4), (23000, 3e-5)),
                      grad_clip=1e-3,
                      n_data_samples=100000, n_val_samples=500,
                      log_freq=5000, summary_freq=10, print_freq=100,
                      prev_ckpt=0,
                      optimizer_name='adam', plot_filters=True, stop_on_overfit=False)

# p = FoEFullPrior('conv1/lin:0', 1e-10, 'alexnet', [hw, hw], 1.0, n_components=2000, n_channels=nchan,
#                  n_features_white=hw**2*nchan, dist='student', mean_mode='gc', sdev_mode='gc', whiten_mode=wmode,
#                  name=None, load_name=None, dir_name=None, load_tensor_names=None)

# make_channel_separate_patch_data(100000, hw, hw, 'alexnet', 'rgb_scaled:0', 3, n_feats_per_channel_white=hw*hw,
#                                  whiten_mode=wmode, batch_size=100, mean_mode='gc', sdev_mode='gc')


# p = FoEChannelwisePrior('rgb_scaled:0', 1e-10, 'alexnet', [hw, hw], 1.0, n_components=150, n_channels=3,
#                         n_features_per_channel_white=hw**2, dist='logistic', mean_mode='gc', sdev_mode='gc',
#                         whiten_mode=wmode,
#                         name=None, load_name=None, dir_name=None, load_tensor_names=None)

# p = FoESeparablePrior('rgb_scaled:0', 1e-10, 'alexnet', [hw, hw], 1.0, n_components=100, n_channels=3,
#                       n_features_per_channel_white=hw**2,
#                       dim_multiplier=10, share_weights=True, channelwise_data=False, loop_loss=True,
#                       dist='logistic', mean_mode='gc', sdev_mode='gc', whiten_mode=wmode,
#                       name=None, load_name=None, dir_name=None, load_tensor_names=None)
#
# p.train_prior(batch_size=500, n_iterations=1000, lr=3e-5,
#               lr_lower_points=((0, 1e-0), (20000, 1e-1),
#                                (21000, 3e-2),
#                                (22000, 1e-2), (22500, 3e-3), (23000, 1e-3),
#                                (24000, 1e-4), (24500, 3e-5)),
#               grad_clip=1e-3,
#               n_data_samples=100000, n_val_samples=1000,
#               log_freq=1000, summary_freq=10, print_freq=10,
#               prev_ckpt=0,
#               optimizer_name='adam', plot_filters=True)

# p.plot_filters_top_alphas(7, p.load_path + 'filter_vis/a_top/')
# p.plot_channels_top_filters(range(7), p.load_path + 'filter_vis/c_top/')

# show_patches_by_channel('/home/frederik/PycharmProjects/deep_restoration/data/patches/image/13x13_mean_gc_sdev_gc_channelwise/',
#                         'raw_mat.npy', 10000, 13, 13, 3, plot_patches=range(10))

# c1l_prior = FoEFullPrior(tensor_names='conv1/lin:0', weighting=1e-10, classifier='alexnet',
#                          filter_dims=[8, 8], input_scaling=1.0, n_components=6000, n_channels=96,
#                          n_features_white=3000, dist='student', mean_mode='gc', sdev_mode='gc',
#                          load_name='FoEPrior')
#
# c1l_prior.train_prior()


# raw_patch_data_mat('conv1/lin:0', 'alexnet', 100, 8, 8, 10, 96,
#                    '../data/patches/alexnet/conv1_lin_8x8_raw/',
#                    file_name='raw_mat.npy')

# raw_mat_dir = '../data/patches/alexnet/conv1_lin_5x5_raw/'
# raw_mat_name = 'raw_mat.npy'
# show_patches_by_channel(raw_mat_dir, raw_mat_name, 100, 5, 5, 96, plot_patches=range(5))

# raw_mat_dir = '../data/patches/alexnet/conv1_relu_8x8_6144feats_mean_gc_sdev_gc/'
# raw_mat_name = 'raw_mat.npy'
# show_patches_by_channel(raw_mat_dir, raw_mat_name, 100000, 8, 8, 96, plot_patches=range(10, 15))


# plot_alexnet_filters('./', filter_name='conv1', filter_ids=(15, 31, 39, 44, 51, 55, 78, 83))

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

# img_prior = FoEPrior(tensor_names='pre_img:0',
#                      weighting=1e-12, name='FoEPrior',
#                      classifier='alexnet',
#                      filter_dims=[8, 8], input_scaling=1.0, n_components=50, n_channels=3,
#                      n_features_white=64*3-1, mean_mode='gc', sdev_mode='gc')

# img_prior.train_prior(batch_size=100, num_iterations=20000,
#                       lr_lower_points=((0, 1e-0), (4000, 1e-1), (6000, 3e-2), (8000, 1e-2),
#                                        (10000, 3e-3), (13000, 1e-3), (15000, 3e-4)),
#                       log_freq=1000, summary_freq=10, print_freq=100,
#                       test_freq=100, n_val_samples=500,
#                       prev_ckpt=0,
#                       optimizer_name='adam')

# c1l_prior = img_prior
# c1l_prior.plot_filters_all_channels(range(5), c1l_prior.load_path + 'filter_vis/')
# c1l_prior.plot_channels_top_filters(range(5), c1l_prior.load_path + 'filter_vis/top/')

# c1l_prior = ChannelICAPrior('conv1_lin:0', 1e-6, 'alexnet', [5, 5], input_scaling=1.0,
#                             n_components=150, n_channels=96,
#                             n_features_white=24,
#                             trainable=False, name='ChannelICAPrior', mean_mode='gc', sdev_mode='gc')

# c1l_prior = FoEPrior(tensor_names='conv1/lin:0',
#                      weighting=1e-12, name='FoEPrior',
#                      classifier='alexnet',
#                      filter_dims=[5, 5], input_scaling=1.0, n_components=6000, n_channels=96,
#                      n_features_white=1800, mean_mode='gc', sdev_mode='gc')

# c1l_prior = FoEPrior(tensor_names='pool1:0',
#                      weighting=1e-12, name='FoEPrior',
#                      classifier='alexnet',
#                      filter_dims=[5, 5], input_scaling=1.0, n_components=4800, n_channels=96,
#                      n_features_white=2400, mean_mode='gc', sdev_mode='gc')
#

# 5x5 flat mean: global channel, sdev: global channel 2400f
# make_flattened_patch_data(num_patches=100000, ph=5, pw=5, classifier='alexnet', map_name='conv1/lin:0',
#                           n_channels=96,
#                           save_dir='../data/patches/alexnet/conv1_lin_5x5_1200feats_mean_lc_sdev_gc/',
#                           n_feats_white=1200, whiten_mode='pca', batch_size=100,
#                           mean_mode='lc', sdev_mode='gc',
#                           raw_mat_load_path='../data/patches/alexnet/new/conv1_lin_5x5_2399feats_mean_gc_sdev_gc/raw_mat.npy')

# add_flattened_validation_set(num_patches=500, ph=8, pw=8, classifier='alexnet', map_name='rgb_scaled:0',
#                              n_channels=3, n_feats_white=64*3-1, whiten_mode='pca', batch_size=100,
#                              mean_mode='global_channel', sdev_mode='global_channel')
