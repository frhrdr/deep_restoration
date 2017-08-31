from utils.preprocessing import make_flattened_patch_data, add_flattened_validation_set
from modules.ica_prior import ICAPrior
from modules.foe_prior import FoEPrior


# #

# make_flattened_patch_data(num_patches=100000, ph=8, pw=8, classifier='alexnet', map_name='conv1/relu:0',
#                           n_channels=96,
#                           n_feats_white=3000, whiten_mode='pca', batch_size=100,
#                           mean_mode='gc', sdev_mode='gc',
#                           raw_mat_load_path='../data/patches/alexnet/conv1_relu_8x8_6144feats_mean_gc_sdev_gc/raw_mat.npy')
#
# add_flattened_validation_set(num_patches=1000, ph=8, pw=8, classifier='alexnet', map_name='conv1/relu:0',
#                              n_channels=96, n_feats_white=3000, whiten_mode='pca', batch_size=100,
#                              mean_mode='global_channel', sdev_mode='global_channel')

prior = FoEPrior(tensor_names='conv1/relu:0',
                 weighting=1e-12, name='FoEPrior',
                 classifier='alexnet',
                 filter_dims=[8, 8], input_scaling=1.0, n_components=6000, n_channels=96,
                 n_features_white=3000, mean_mode='gc', sdev_mode='gc')

prior.train_prior(batch_size=500, num_iterations=24000, lr=3e-5,
                  lr_lower_points=((0, 1e-0), (6000, 1e-1), (6500, 3e-2),
                                       (7000, 1e-2), (8000, 3e-3), (9000, 1e-3),
                                       (11000, 3e-4), (12000, 1e-4), (13000, 1e-5)),
                  grad_clip=100.0,
                  whiten_mode='pca', num_data_samples=100000,
                  log_freq=1000, summary_freq=10, print_freq=100,
                  test_freq=100, n_val_samples=1000,
                  prev_ckpt=6000,
                  optimizer_name='adam',
                  plot_filters=False, do_clip=True)

# prior.plot_filters_all_channels(range(5), prior.load_path + 'filter_vis/')
# prior.plot_channels_top_filters(range(5), prior.load_path + 'filter_vis/top/')

# foe_prior = FoEPrior(tensor_names='conv2/lin:0',
#                      weighting=1e-9, name='FoEPrior',
#                      classifier='alexnet',
#                      filter_dims=[5, 5], input_scaling=1.0, n_components=6400, n_channels=256,
#                      n_features_white=3200, mean_mode='lc', sdev_mode='gc')

# foe_prior.plot_filters(range(3), foe_prior.load_path)

# foe_prior.train_prior(batch_size=500, num_iterations=30000, lr=3e-5,
#                       lr_lower_points=((0, 1e-0), (3000, 1e-1), (6000, 3e-2), (16000, 1e-2),
#                                        (21000, 3e-3), (26000, 1e-3), (31000, 3e-4), (36000, 1e-4)),
#                       grad_clip=100.0,
#                       whiten_mode='pca', num_data_samples=100000,
#                       log_freq=1000, summary_freq=10, print_freq=100,
#                       prev_ckpt=0, optimizer_name='adam',
#                       plot_filters=False, do_clip=True)