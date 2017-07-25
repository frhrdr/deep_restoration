from modules.ica_prior import ICAPrior

# make_feat_map_mats(num_patches=100000, map_name='conv2/relu:0', classifier='alexnet', ph=5, pw=5,
#                    save_dir='../data/patches/alexnet/conv2_relu_5x5/', whiten_mode='pca', batch_size=50)
#
# make_reduced_feat_map_mats(100000, load_dir='../data/patches/alexnet/conv2_relu_5x5/',
#                            n_features=6400, n_to_keep=3200,
#                            save_dir='../data/patches/alexnet/conv2_relu_5x5_3200feats/')

# kept 97.8% eigv fraction

ica_prior = ICAPrior(tensor_names='conv2/relu:0',
                     weighting=0.00001, name='ICAPrior',
                     load_path='../logs/priors/ica_prior/alexnet/5x5_conv2_relu_6400comp_3200feats/',
                     trainable=False, filter_dims=[5, 5], input_scaling=1.0, n_components=2000, n_channels=256,
                     n_features_white=3200)

ica_prior.train_prior(batch_size=200, num_iterations=30000,
                      lr_lower_points=[(0, 1.0e-4), (10000, 1.0e-5), (20000, 3.0e-6), (25000, 1.0e-6)],
                      whiten_mode='pca', data_dir='../data/patches/alexnet/conv2_relu_5x5_3200feats/',
                      num_data_samples=100000, n_features=1000,
                      plot_filters=False)