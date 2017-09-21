from net_inversion import NetInversion
from modules.foe_full_prior import FoEFullPrior
from modules.foe_channelwise_prior import FoEChannelwisePrior
from modules.foe_separable_prior import FoESeparablePrior
from modules.loss_modules import MSELoss, TotalVariationLoss
from modules.split_module import SplitModule
from shutil import copyfile
import os
import numpy as np

split1 = SplitModule(name_to_split='conv1/lin:0', img_slice_name='img_rep_c1l',
                     rec_slice_name='conv1_lin', name='Split1')
mse1 = MSELoss(target='img_rep_c1l:0', reconstruction='conv1_lin:0', name='MSE_conv1_tracker')
mse1.add_loss = False

split2 = SplitModule(name_to_split='conv2/lin:0', img_slice_name='img_rep_c2l',
                     rec_slice_name='rec_rep_c2l', name='Split2')
mse2 = MSELoss(target='img_rep_c2l:0', reconstruction='rec_rep_c2l:0', name='MSE_conv2')
mse2.add_loss = True

split3 = SplitModule(name_to_split='conv3/lin:0', img_slice_name='img_rep_c3l',
                     rec_slice_name='rec_rep_c3l', name='Split3')
# mse3 = MSELoss(target='img_rep_c3l:0', reconstruction='rec_rep_c3l:0', name='MSE_conv3')
mse3 = MSELoss(target='img_rep_c3l:0', reconstruction='rec_rep_c3l:0', name='MSE_Reconstruction')
mse3.add_loss = False

split4 = SplitModule(name_to_split='conv4/lin:0', img_slice_name='img_rep_c4l',
                     rec_slice_name='rec_rep_c4l', name='Split4')
mse4 = MSELoss(target='img_rep_c4l:0', reconstruction='rec_rep_c4l:0', name='MSE_c4l')
mse4.add_loss = True

split5 = SplitModule(name_to_split='conv5/lin:0', img_slice_name='img_rep_c5l',
                     rec_slice_name='rec_rep_c5l', name='Split5')
mse5 = MSELoss(target='img_rep_c5l:0', reconstruction='rec_rep_c5l:0', name='MSE_Reconstruction')
mse5.add_loss = False

split6 = SplitModule(name_to_split='fc6/lin:0', img_slice_name='img_rep_fc6l',
                     rec_slice_name='rec_rep_fc6l', name='Split6')
mse6 = MSELoss(target='img_rep_fc6l:0', reconstruction='rec_rep_fc6l:0', name='MSE_fc6l')
mse6.add_loss = True

pre_mse = MSELoss(target='target_featmap/read:0', reconstruction='pre_featmap/read:0', name='MSE_Image')
pre_mse.add_loss = False

# fullprior = FoEFullPrior(tensor_names='pre_featmap/read:0', weighting=1e-8, classifier='alexnet',
#                          filter_dims=[8, 8], input_scaling=1.0, n_components=6000, n_channels=96,
#                          n_features_white=3000, dist='student', mean_mode='gc', sdev_mode='gc',
#                          load_name='FoEPrior',
#                          load_tensor_names='conv1/lin:0')

slimprior = FoEFullPrior('pre_featmap/read:0', 1e-9, 'alexnet', [3, 3], 1.0, n_components=7000, n_channels=384,
                         n_features_white=3**2*384, dist='student', mean_mode='gc', sdev_mode='gc', whiten_mode='pca',
                         name=None, load_name=None, dir_name=None, load_tensor_names='conv3/lin:0')

# chanprior = FoEChannelwisePrior(tensor_names='pre_featmap/read:0', weighting=6e-5, classifier='alexnet',
#                                 filter_dims=[8, 8], input_scaling=1.0, n_components=150, n_channels=96,
#                                 n_features_per_channel_white=64,
#                                 dist='logistic', mean_mode='gc', sdev_mode='gc', whiten_mode='zca',
#                                 load_tensor_names='conv1/lin:0')

imgprior = FoEFullPrior('pre_featmap/read:0', 1e-5, 'alexnet', [12, 12], 1.0, n_components=1000, n_channels=3,
                        n_features_white=12**2*3, dist='student', mean_mode='gc', sdev_mode='gc', whiten_mode='pca',
                        name=None, load_name=None, dir_name=None, load_tensor_names='image')

p = FoESeparablePrior('rgb_scaled:0', 1e-10, 'alexnet', [9, 9], 1.0, n_components=500, n_channels=3,
                      n_features_per_channel_white=9**2,
                      dim_multiplier=50, share_weights=False, channelwise_data=True,
                      dist='student', mean_mode='gc', sdev_mode='gc', whiten_mode='zca',
                      name=None, load_name=None, dir_name=None, load_tensor_names=None)


tv_prior = TotalVariationLoss(tensor='pre_featmap/read:0', beta=2, weighting=1e-10)

modules = [split6, mse6, split5, mse5, imgprior, pre_mse]
log_path = '../logs/opt_inversion/alexnet/slim_vs_img/fc6l_to_c5l/pre_image_12x12_full_prior/1e-5/'
# log_path = '../logs/opt_inversion/alexnet/sep_prior_on_img/channelwise/'
ni = NetInversion(modules, log_path, classifier='alexnet', summary_freq=10, print_freq=50, log_freq=500)

if not os.path.exists(log_path):
    os.makedirs(log_path)
copyfile('./opt_inv_script.py', log_path + 'script.py')

# pre_img_init = np.reshape(np.load(params['log_path'] + 'mats/rec_10500.npy'), [1, 224, 224, 3])

# pre_featmap_init = np.load('../logs/opt_inversion/alexnet/pre_featmap/mse_init_1500.npy')
# pre_img_init = np.load('../logs/net_inversion/alexnet/c1l_tests_16_08/init_helper.npy')

pre_featmap_init = None

ni.train_pre_featmap('../data/selected/images_resized_227/red-fox.bmp', n_iterations=500, optim_name='adam',
                     lr_lower_points=((1e+0, 3e-1),), grad_clip=10000.,
                     pre_featmap_init=pre_featmap_init, ckpt_offset=0,
                     pre_featmap_name='rgb_scaled',
                     featmap_names_to_plot=(), max_n_featmaps_to_plot=10, save_as_plot=False)

pre_featmap_init = np.load(ni.log_path + 'mats/rec_500.npy')

ni.train_pre_featmap('../data/selected/images_resized_227/red-fox.bmp', n_iterations=9500, optim_name='adam',
                     lr_lower_points=((1e+0, 3e-1),), grad_clip=10000.,
                     pre_featmap_init=pre_featmap_init, ckpt_offset=500,
                     pre_featmap_name='rgb_scaled',
                     featmap_names_to_plot=(), max_n_featmaps_to_plot=10, save_as_plot=False)
