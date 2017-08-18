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

c1l_prior = ChannelICAPrior('conv1_lin:0', 1e-7, 'alexnet', [5, 5], input_scaling=1.0,
                            n_components=150, n_channels=96,
                            n_features_white=24,
                            trainable=False, name='ChannelICAPrior', mean_mode='gc', sdev_mode='gc')

modules = [split4, mse4, split1, mse1, c1l_prior]

params = dict(classifier='alexnet',
              modules=modules,
              log_path='../logs/net_inversion/alexnet/c1l_tests_16_08/6_MSE_c4l_CICA_c1l/1e-7/',
              load_path='')
params.update(mv_default_params())
params['num_iterations'] = 10000
params['learning_rate'] = 1e-10

if not os.path.exists(params['log_path']):
    os.makedirs(params['log_path'])
copyfile('./ni_tests.py', params['log_path'] + 'script.py')

ni = NetInversion(params)

pre_img_init = np.reshape(np.load(params['log_path'] + 'mats/rec_10000.npy'), [1, 224, 224, 3])
# pre_img_init = np.load('../logs/net_inversion/alexnet/c1l_tests_16_08/init_helper.npy')

ni.train_pre_image('../data/selected/images_resized/red-fox.bmp', optim_name='adam',
                   jitter_t=0, jitter_stop_point=0, range_clip=False, scale_pre_img=1.0,
                   lr_lower_points=((0, 1e-2),), grad_clip=10000.,
                   save_as_plot=False, pre_img_init=pre_img_init, ckpt_offset=10000,
                   featmap_names_to_plot=('conv2/lin:0',), max_n_featmaps_to_plot=10)
