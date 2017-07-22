# from utils.temp_utils import make_feat_map_mats, make_reduced_feat_map_mats
from modules.ica_prior import ICAPrior

# make_feat_map_mats(num_patches=100000, map_name='conv3/relu:0', classifier='alexnet', ph=5, pw=5,
#                    save_dir='../data/patches/alexnet/conv3_relu_5x5/', whiten_mode='pca', batch_size=10)

# make_reduced_feat_map_mats(100000, load_dir='../data/patches/alexnet/conv3_relu_5x5/',
#                            n_features=9600, n_to_keep=1000,
#                            save_dir='../data/patches/alexnet/conv3_relu_5x5_1000feats/')

ica_prior = ICAPrior(tensor_names='conv3/relu:0',
                     weighting=0.00001, name='ICAPrior',
                     load_path='../logs/priors/ica_prior/alexnet/5x5_conv3_relu_20kcomp_10kfeats/',
                     trainable=False, filter_dims=[5, 5], input_scaling=1.0, n_components=10000, n_channels=384,
                     n_features_white=9599)

ica_prior.train_prior(batch_size=500, num_iterations=50000,
                      lr_lower_points=[(0, 1.0e-4), (10000, 1.0e-5), (40000, 3.0e-6), (45000, 1.0e-6)],
                      whiten_mode='pca', data_dir='../data/patches/alexnet/conv3_relu_5x5/',
                      num_data_samples=100000, n_features=9599,
                      plot_filters=False)
