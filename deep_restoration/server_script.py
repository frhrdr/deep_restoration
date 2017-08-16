from utils.preprocessing import make_flattened_patch_data
from modules.ica_prior import ICAPrior
from modules.foe_prior import FoEPrior

# make_flattened_patch_data(num_patches=100000, ph=5, pw=5, classifier='alexnet', map_name='conv2/lin:0',
#                           n_channels=256,
#                           save_dir='../data/patches/alexnet/conv2_lin_5x5_2000feats_mean_lc_sdev_gc/',
#                           n_feats_white=2000, whiten_mode='pca', batch_size=100,
#                           mean_mode='lc', sdev_mode='gc')

foe_prior = FoEPrior(tensor_names='conv1/lin:0',
                     weighting=1e-9, name='FoEPrior',
                     classifier='alexnet',
                     filter_dims=[5, 5], input_scaling=1.0, n_components=3000, n_channels=96,
                     n_features_white=1800, mean_mode='lc', sdev_mode='gc')

foe_prior.train_prior(batch_size=500, num_iterations=30000, lr=3e-5,
                      lr_lower_points=((0, 3e-2), (10000, 1e-2), (15000, 3e-3), (20000, 1e-3), (25000, 3e-4), (30000, 1e-4)),
                      grad_clip=100.0,
                      whiten_mode='pca', num_data_samples=100000,
                      log_freq=1000, summary_freq=10, print_freq=100,
                      prev_ckpt=6000, optimizer_name='adam',
                      plot_filters=False, do_clip=True)