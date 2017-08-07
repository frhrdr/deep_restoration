from utils.temp_utils import make_channel_separate_feat_map_mats, make_feat_map_mats, make_reduced_feat_map_mats
from modules.channel_ica_prior import ChannelICAPrior
from modules.ica_prior import ICAPrior
from modules.foe_prior import FoEPrior
import numpy as np
# r = range(500, 17000, 500)
# for i in r:
#     mat_to_img('../logs/net_inversion/alexnet/c4_rec/mse4_pimg_pc2rich_1e-9/mats/rec_' + str(i) + '.npy', cols=1)
# mat_to_img('../logs/net_inversion/alexnet/c3_rec/mse3_foe2lin_1e-8/mats/rec_10000.npy', cols=1)

# make_channel_separate_feat_map_mats(num_patches=100000, ph=8, pw=8, classifier='alexnet',
#                                     map_name='conv1/lin:0', n_channels=96,
#                                     save_dir='../data/patches/alexnet/conv1_lin_8x8_63feats_channelwise/',
#                                     whiten_mode='pca', batch_size=100)
#
make_feat_map_mats(100000, map_name='conv2/lin:0', classifier='alexnet', ph=5, pw=5,
                   save_dir='../data/patches/alexnet/conv2_lin_5x5_6399feats_redo/',
                   whiten_mode='pca', batch_size=100)

make_feat_map_mats(100000, map_name='conv2/relu:0', classifier='alexnet', ph=5, pw=5,
                   save_dir='../data/patches/alexnet/conv2_relu_5x5_6399feats_redo/',
                   whiten_mode='pca', batch_size=100)
#
# make_reduced_feat_map_mats(num_patches=100000, load_dir='../data/patches/alexnet/conv1_lin_5x5_2399feats/',
#                            n_features=2399, n_to_keep=1200,
#                            save_dir='../data/patches/alexnet/conv1_lin_5x5_1200feats/', whiten_mode='pca')

# c1_prior = ICAPrior(tensor_names='conv1/lin:0',
#                     weighting=1., name='FoEPrior',
#                     classifier='alexnet',
#                     filter_dims=[5, 5], input_scaling=1.0, n_components=4800, n_channels=96,
#                     n_features_white=2399)
#
# c1_prior.train_prior(batch_size=250, num_iterations=30000, lr=1e-7,
#                     lr_lower_points=((0, 3e-7), (5000, 3e-4), (10000, 3e-4), (25000, 3e-5)),
#                     grad_clip=100.0,
#                     whiten_mode='pca', num_data_samples=100000,
#                     log_freq=5000, summary_freq=10, print_freq=1, prev_ckpt=0, optimizer_name='adam',
#                     plot_filters=False, do_clip=True)









# from mahendran_vedaldi_2016 import invert_layer
#
# params = dict(image_path='./data/selected/images_resized/val13_monkey.bmp', layer_name='conv3/relu:0',
#               classifier='alexnet',
#               img_HW=224,
#               jitter_T=0,
#               mse_C=300.0,
#               alpha_sr=6, beta_tv=2,
#               range_B=80,
#               range_V=80/6.5,
#               learning_rate=0.1,
#               num_iterations=10000,
#               print_freq=100, log_freq=2000, summary_freq=10, lr_lower_freq=10000,
#               grad_clip=100.0,
#               log_path='./logs/mahendran_vedaldi/2016/',
#               save_as_mat=False)
#
# invert_layer(params)

# from net_inversion import NetInversion
# from parameter_utils import default_params
#
#
# spec1 = dict(inv_model_type='deconv_conv',
#              op1_height=5, op1_width=5, op1_strides=[1, 2, 2, 1], op1_pad='SAME',
#              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1], op2_pad='SAME',
#              hidden_channels=256, target_shape=[28, 28, 256],
#              inv_input_name='pool3:0', inv_target_name='conv3_3/relu:0',
#              rec_name='conv3_3_rec', add_loss=True)
#
# spec2 = dict(inv_model_type='deconv_conv',
#              op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1], op1_pad='SAME',
#              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1], op2_pad='SAME',
#              hidden_channels=256, target_shape=[56, 56, 256],
#              inv_input_name='module_1/conv3_3_rec:0', inv_target_name='conv3_2/relu:0',
#              rec_name='conv3_2_rec', add_loss=True)
#
# spec3 = dict(inv_model_type='deconv_conv',
#              op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1], op1_pad='SAME',
#              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1], op2_pad='SAME',
#              hidden_channels=256, target_shape=[56, 56, 256],
#              inv_input_name='module_2/conv3_2_rec:0', inv_target_name='conv3_1/relu:0',
#              rec_name='conv3_1_rec', add_loss=True)
#
# spec4 = dict(inv_model_type='deconv_conv',
#              op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1], op1_pad='SAME',
#              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1], op2_pad='SAME',
#              hidden_channels=256, target_shape=[56, 56, 128],
#              inv_input_name='module_3/conv3_1_rec:0', inv_target_name='pool2:0',
#              rec_name='pool2_rec', add_loss=True)
#
# spec5 = dict(inv_model_type='deconv_conv',
#              op1_height=5, op1_width=5, op1_strides=[1, 2, 2, 1], op1_pad='SAME',
#              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1], op2_pad='SAME',
#              hidden_channels=128, target_shape=[112, 112, 128],
#              inv_input_name='module_4/pool2_rec:0', inv_target_name='conv2_2/relu:0',
#              rec_name='conv2_2_rec', add_loss=True)
#
# spec6 = dict(inv_model_type='deconv_conv',
#              op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1], op1_pad='SAME',
#              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1], op2_pad='SAME',
#              hidden_channels=128, target_shape=[112, 112, 128],
#              inv_input_name='module_5/conv2_2_rec:0', inv_target_name='conv2_1/relu:0',
#              rec_name='conv2_1_rec', add_loss=True)
#
# spec7 = dict(inv_model_type='deconv_conv',
#              op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1], op1_pad='SAME',
#              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1], op2_pad='SAME',
#              hidden_channels=128, target_shape=[112, 112, 64],
#              inv_input_name='module_6/conv2_1_rec:0', inv_target_name='pool1:0',
#              rec_name='pool1_rec', add_loss=True)
#
# spec8 = dict(inv_model_type='deconv_conv',
#              op1_height=5, op1_width=5, op1_strides=[1, 2, 2, 1], op1_pad='SAME',
#              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1], op2_pad='SAME',
#              hidden_channels=64, target_shape=[224, 224, 64],
#              inv_input_name='module_7/pool1_rec:0', inv_target_name='conv1_2/relu:0',
#              rec_name='conv1_2_rec', add_loss=True)
#
# spec9 = dict(inv_model_type='deconv_conv',
#              op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1], op1_pad='SAME',
#              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1], op2_pad='SAME',
#              hidden_channels=64, target_shape=[224, 224, 64],
#              inv_input_name='module_8/conv1_2_rec:0', inv_target_name='conv1_1/relu:0',
#              rec_name='conv1_1_rec', add_loss=True)
#
# spec10 = dict(inv_model_type='deconv_conv',
#               op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1], op1_pad='SAME',
#               op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1], op2_pad='SAME',
#               hidden_channels=64, target_shape=[224, 224, 3],
#               inv_input_name='module_9/conv1_1_rec:0', inv_target_name='bgr_normed:0',
#               rec_name='reconstruction', add_loss=True)
#
# params = dict(classifier='vgg16',
#               inv_model_specs=[spec1, spec2, spec3, spec4, spec5, spec6, spec7, spec8, spec9, spec10],
#               log_path='./logs/net_inversion/vgg16/l1-10_10dc/run6/',
#               load_path='')
# params.update(default_params())
# params['learning_rate'] = 3.0e-5
# print(params)
# ni = NetInversion(params)
# ni.visualize()
#
