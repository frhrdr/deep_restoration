from net_inversion import NetInversion
from modules.ica_prior import ICAPrior
from modules.foe_prior import FoEPrior
from modules.channel_ica_prior import ChannelICAPrior
from modules.loss_modules import NormedMSELoss
from modules.split_module import SplitModule
from utils.parameter_defaults import mv_default_params
from shutil import copyfile
import os
import numpy as np

split2 = SplitModule(name_to_split='conv2/lin:0', img_slice_name='img_rep_c2l',
                     rec_slice_name='rec_rep_c2l', name='Split2')
mse2 = NormedMSELoss(target='img_rep_c2l:0', reconstruction='rec_rep_c2l:0', name='MSE_conv2')
mse2.add_loss = True

split4 = SplitModule(name_to_split='conv4/lin:0', img_slice_name='img_rep_c4l',
                     rec_slice_name='rec_rep_c4l', name='Split4')
mse4 = NormedMSELoss(target='img_rep_c4l:0', reconstruction='rec_rep_c4l:0', name='MSE_c4l')
mse4.add_loss = True

split1 = SplitModule(name_to_split='conv1/lin:0', img_slice_name='img_rep_c1l',
                     rec_slice_name='conv1_lin', name='Split1')
mse1 = NormedMSELoss(target='img_rep_c1l:0', reconstruction='conv1_lin:0', name='MSE_conv1_tracker')
mse1.add_loss = False

# split1 = SplitModule(name_to_split='conv1/relu:0', img_slice_name='img_rep_c1r',
#                      rec_slice_name='conv1_relu', name='Split1')
# mse1 = NormedMSELoss(target='img_rep_c1r:0', reconstruction='conv1_relu:0', name='MSE_conv1_tracker')
# mse1.add_loss = False

img_prior = ICAPrior(tensor_names='pre_img/read:0',
                     weighting=1e-7, name='ICAPrior',
                     classifier='alexnet',
                     filter_dims=[8, 8], input_scaling=1.0, n_components=512, n_channels=3,
                     n_features_white=64*3-1,
                     mean_mode='lf', sdev_mode='gc')

# c1l_prior = ICAPrior(tensor_names='conv1_lin:0',
#                      weighting=1e-13, name='C1LPrior',
#                      classifier='alexnet',
#                      filter_dims=[5, 5], input_scaling=1.0, n_components=3000, n_channels=96,
#                      n_features_white=1800, mean_mode='lc', sdev_mode='gc')

# c1l_prior = FoEPrior(tensor_names='conv1_lin:0',
#                      weighting=1e-12, name='FoEC1LPrior',
#                      classifier='alexnet',
#                      filter_dims=[5, 5], input_scaling=1.0, n_components=3000, n_channels=96,
#                      n_features_white=1800, mean_mode='lc', sdev_mode='gc')

# c1l_prior = ChannelICAPrior('pre_featmap/read:0', 1e-6, 'alexnet', [5, 5], input_scaling=1.0,
#                             n_components=150, n_channels=96,
#                             n_features_white=24,
#                             trainable=False, name='ChannelICAPrior', mean_mode='gc', sdev_mode='gc',
#                             load_tensor_names='conv1/lin:0')

c1l_prior = FoEPrior(tensor_names='pre_featmap/read:0',
                     weighting=1e-10, name='FoEPrior',
                     classifier='alexnet',
                     filter_dims=[8, 8], input_scaling=1.0, n_components=6000, n_channels=96,
                     n_features_white=3000, mean_mode='gc', sdev_mode='gc',
                     load_tensor_names='conv1/lin:0')

pre_mse = NormedMSELoss(target='target_featmap/read:0', reconstruction='pre_featmap/read:0', name='MSE_Reconstruction')
pre_mse.add_loss = False

# modules = [split4, mse4, split1, mse1, c1l_prior]
# modules = [split4, mse4, split1, mse1, c1l_prior]

modules = [split4, mse4, pre_mse, c1l_prior]
log_path='../logs/opt_inversion/alexnet/pre_featmap/88/1e-10/'

ni = NetInversion(modules, log_path, classifier='alexnet', summary_freq=10, print_freq=50, log_freq=500)

if not os.path.exists(log_path):
    os.makedirs(log_path)
copyfile('./ni_tests.py', log_path + 'script.py')


# pre_img_init = np.reshape(np.load(params['log_path'] + 'mats/rec_10500.npy'), [1, 224, 224, 3])

pre_featmap_init = np.load('../logs/opt_inversion/alexnet/pre_featmap/mse_init_1500.npy')

# pre_img_init = np.load('../logs/net_inversion/alexnet/c1l_tests_16_08/init_helper.npy')

# pre_featmap_init = None

# pre_featmap_init = np.random.normal(loc=0, scale=0.1, size=(1, 56, 56, 96)).astype(np.float32)

ni.train_pre_featmap('../data/selected/images_resized/red-fox.bmp', n_iterations=10000, optim_name='adam',
                     lr_lower_points=((1e+0, 3e-1),), grad_clip=10000.,
                     pre_featmap_init=pre_featmap_init, ckpt_offset=0,
                     pre_featmap_name = 'conv1/lin',
                     featmap_names_to_plot=(), max_n_featmaps_to_plot=10)

# pre_featmap_init = np.reshape(np.load(params['log_path'] + 'mats/rec_500.npy'), [1, 56, 56, 96])
# ni.params['num_iterations'] = 10000
# ni.train_pre_image('../data/selected/images_resized/red-fox.bmp', optim_name='adam',
#                    jitter_t=0, jitter_stop_point=0, range_clip=False, scale_pre_img=1.0,
#                    lr_lower_points=((1e+0, 3e-1),), grad_clip=10000.,
#                    pre_featmap_init=pre_featmap_init, ckpt_offset=500,
#                    pre_featmap_name = 'conv1/lin',
#                    featmap_names_to_plot=(), max_n_featmaps_to_plot=10)