from foe_inv_funcs import featmap_inv

match_layer = 4
target_layer = 1
jitter_t = 0
weighting = '1e-4'
restart_adam = False
image_name = 'val153'
do_plot = True


prior_id = 'fullc1l6000'
custom_target = None
pre_image = False
make_mse = True
featmap_inv(match_layer, target_layer, image_name, prior_id, prior_weighting=weighting, make_mse=make_mse,
            restart_adam=restart_adam, pre_image=pre_image, do_plot=do_plot,
            jitter_t=0, jitter_stop_point=3200, lr=1., bound_plots=True, custom_target=custom_target)
make_mse = False
featmap_inv(match_layer, target_layer, image_name, prior_id, prior_weighting=weighting, make_mse=make_mse,
            restart_adam=restart_adam, pre_image=pre_image, do_plot=do_plot,
            jitter_t=0, jitter_stop_point=3200, lr=1., bound_plots=True, custom_target=custom_target)

# custom_target = None
# pre_image = True
# make_mse = True
# featmap_inv(match_layer, target_layer, image_name, prior_id, prior_weighting=weighting, make_mse=make_mse,
#             restart_adam=restart_adam, pre_image=pre_image, do_plot=do_plot,
#             jitter_t=0, jitter_stop_point=3200, lr=1., bound_plots=True)
# make_mse = False
# featmap_inv(match_layer, target_layer, image_name, prior_id, prior_weighting=weighting, make_mse=make_mse,
#             restart_adam=restart_adam, pre_image=pre_image, do_plot=do_plot,
#             jitter_t=0, jitter_stop_point=3200, lr=1., bound_plots=True)


# resize_all_images((227, 227), 'images_resized_227')

# make_flattened_patch_data(num_patches=100000, ph=3, pw=3, classifier='alexnet', map_name='conv2/lin:0',
#                           n_channels=256,
#                           n_feats_white=9*256, whiten_mode='pca', batch_size=100,
#                           mean_mode='gc', sdev_mode='gc',
#                           raw_mat_load_path='', n_val_patches=1000)
#
# make_flattened_patch_data(num_patches=100000, ph=3, pw=3, classifier='alexnet', map_name='conv3/lin:0',
#                           n_channels=384,
#                           n_feats_white=9*384, whiten_mode='pca', batch_size=100,
#                           mean_mode='gc', sdev_mode='gc',
#                           raw_mat_load_path='', n_val_patches=1000)
#
# make_flattened_patch_data(num_patches=100000, ph=3, pw=3, classifier='alexnet', map_name='conv4/lin:0',
#                           n_channels=384,
#                           n_feats_white=9*384, whiten_mode='pca', batch_size=100,
#                           mean_mode='gc', sdev_mode='gc',
#                           raw_mat_load_path='', n_val_patches=1000)
#
# make_flattened_patch_data(num_patches=100000, ph=3, pw=3, classifier='alexnet', map_name='conv5/lin:0',
#                           n_channels=256,
#                           n_feats_white=9*256, whiten_mode='pca', batch_size=100,
#                           mean_mode='gc', sdev_mode='gc',
#                           raw_mat_load_path='', n_val_patches=1000)

# hw = 3
# wmode = 'pca'
#
# params = [(2, 256, 5000), (5, 256, 5000), (3, 384, 7000), (4, 384, 7000)]
# for layer, nchan, ncomp in params:
#     p = FoEFullPrior('conv{}/lin:0'.format(layer), 1e-10, 'alexnet', [hw, hw], 1.0, n_components=ncomp,
#                      n_channels=nchan,
#                      n_features_white=hw**2*nchan, dist='student', mean_mode='gc', sdev_mode='gc', whiten_mode=wmode,
#                      name=None, load_name=None, dir_name=None, load_tensor_names=None)
#
#     p.train_prior(batch_size=500, n_iterations=20000,
#                   lr_lower_points=((0, 1e-0), (10000, 1e-1), (12000, 3e-2),
#                                    (13000, 1e-2), (14500, 3e-3), (15000, 1e-3),
#                                    (16000, 1e-4), (18000, 3e-4), (19000, 1e-5)),
#                   grad_clip=100.0,
#                   n_data_samples=100000,
#                   log_freq=1000, summary_freq=10, print_freq=100, test_freq=100, n_val_samples=1000,
#                   prev_ckpt=0,
#                   optimizer_name='adam')




# make_channel_separate_patch_data(num_patches=100000, ph=8, pw=8, classifier='alexnet', map_name='conv1/lin:0',
#                                  n_channels=96, n_feats_per_channel_white=8*8, whiten_mode='zca', batch_size=100,
#                                  mean_mode='gc', sdev_mode='gc',
#                                  raw_mat_load_path='')



# prior = FoEFullPrior(tensor_names='conv1/lin:0', weighting=1e-10, classifier='alexnet',
#                      filter_dims=[8, 8], input_scaling=1.0, n_components=6000, n_channels=96,
#                      n_features_white=3000, dist='student', mean_mode='gc', sdev_mode='gc',
#                      load_name='FoEPrior')
# #


# prior.plot_filters_all_channels(range(7), prior.load_path + 'filter_vis/')
# prior.plot_channels_top_filters(range(7), prior.load_path + 'filter_vis/top/')
