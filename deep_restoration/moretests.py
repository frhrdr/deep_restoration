from utils.preprocessing import make_channel_separate_patch_data, make_flattened_patch_data

make_channel_separate_patch_data(num_patches=1000, ph=5, pw=5, classifier='alexnet', map_name='rgb_scaled:0',
                                 n_channels=3,
                                 save_dir='../data/patches/alexnet/new/conv1_lin_5x5_channelwise/',
                                 whiten_mode='pca', batch_size=100,
                                 mean_mode='local_full', cov_mode='global_channel')
#
#
make_flattened_patch_data(num_patches=1000, ph=5, pw=5, classifier='alexnet', map_name='rgb_scaled:0',
                          n_channels=3,
                          save_dir='../data/patches/alexnet/new/conv1_lin_5x5_74feats/',
                          n_feats_white=74, whiten_mode='pca', batch_size=100,
                          mean_mode='local_full', cov_mode='global_feature')
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
# make_reduced_feat_map_mats(num_patches=100000, load_dir='../data/patches/alexnet/conv1_lin_5x5_2399feats/',
#                            n_features=2399, n_to_keep=1200,
#                            save_dir='../data/patches/alexnet/conv1_lin_5x5_1200feats/', whiten_mode='pca')


#
# cica_prior = ChannelICAPrior(tensor_names='conv1/lin:0',
#                              weighting=1e-9, name='ChannelICAPrior',
#                              classifier='alexnet',
#                              filter_dims=[8, 8], input_scaling=1.0, n_components=150, n_channels=96,
#                              n_features_white=63)
#
# cica_prior.train_prior(batch_size=500, num_iterations=30000, lr=3e-5,
#                        lr_lower_points=((0, 3e-5),),
#                        grad_clip=100.0, n_vis=144,
#                        whiten_mode='pca', num_data_samples=100000,
#                        log_freq=5000, summary_freq=10, print_freq=100, prev_ckpt=80000, optimizer_name='adam',
#                        plot_filters=False, do_clip=True)
