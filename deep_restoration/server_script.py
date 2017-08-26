from utils.preprocessing import make_flattened_patch_data, add_flattened_validation_set
from modules.ica_prior import ICAPrior
from modules.foe_prior import FoEPrior

# make_flattened_patch_data(num_patches=100000, ph=5, pw=5, classifier='alexnet', map_name='pool1:0',
#                           n_channels=96,
#                           save_dir='../data/patches/alexnet/pool1_5x5_2400feats_mean_gc_sdev_gc/',
#                           n_feats_white=2400, whiten_mode='pca', batch_size=100,
#                           mean_mode='gc', sdev_mode='gc')
# #
# make_flattened_patch_data(num_patches=100000, ph=5, pw=5, classifier='alexnet', map_name='conv2/lin:0',
#                           n_channels=256,
#                           save_dir='../data/patches/alexnet/conv2_lin_5x5_3200feats_mean_gc_sdev_gc/',
#                           n_feats_white=3200, whiten_mode='pca', batch_size=100,
#                           mean_mode='gc', sdev_mode='gc',
#                           raw_mat_load_path='')

# add_flattened_validation_set(num_patches=1000, ph=5, pw=5, classifier='alexnet', map_name='conv2/lin:0',
#                              n_channels=256, n_feats_white=3200, whiten_mode='pca', batch_size=100,
#                              mean_mode='global_channel', sdev_mode='global_channel')

c1l_prior = FoEPrior(tensor_names='conv2/lin:0',
                     weighting=1e-12, name='FoEPrior',
                     classifier='alexnet',
                     filter_dims=[5, 5], input_scaling=1.0, n_components=1000, n_channels=256,
                     n_features_white=3200, mean_mode='gc', sdev_mode='gc')

c1l_prior.train_prior(batch_size=500, num_iterations=40000, lr=3e-5,
                      lr_lower_points=((0, 1e-0), (7000, 1e-1), (9000, 3e-2),
                                       (10000, 1e-2), (12000, 3e-3), (14000, 1e-3),
                                       (17000, 3e-4), (18000, 1e-4), (20000, 1e-5),
                                       (30000, 3e-4), (32000, 1e-4), (34000, 1e-5), (36000, 3e-6), (38000, 1e-6),
                                       (40000, 3e-7), (41000, 1e-7), (42000, 3e-8), (43000, 1e-8)),
                      grad_clip=100.0,
                      whiten_mode='pca', num_data_samples=100000,
                      log_freq=1000, summary_freq=10, print_freq=100,
                      test_freq=100, n_val_samples=1000,
                      prev_ckpt=10000,
                      optimizer_name='adam',
                      plot_filters=False, do_clip=True)

# c1l_prior.plot_filters_all_channels(range(5), c1l_prior.load_path + 'filter_vis/')
# c1l_prior.plot_channels_top_filters(range(5), c1l_prior.load_path + 'filter_vis/top/')

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