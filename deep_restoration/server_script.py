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
mse2 = NormedMSELoss(target='img_rep_c2l:0', reconstruction='rec_rep_c2l:0',
                     weighting=1., name='MSE_conv2')

split1 = SplitModule(name_to_split='conv1/lin:0', img_slice_name='img_rep_c1l',
                     rec_slice_name='conv1_lin', name='Split1')
mse1 = NormedMSELoss(target='img_rep_c1l:0', reconstruction='conv1_lin:0',
                     weighting=1., name='MSE_conv1_tracker')
mse1.add_loss = False

# ica_prior = ICAPrior(tensor_names='pool1:0',
#                      weighting=0.001, name='ICAPrior',
#                      load_path='../logs/priors/ica_prior/alexnet/pool1_3x3/ckpt-20000',
#                      trainable=False, filter_dims=[3, 3], input_scaling=1.0, n_components=512, n_channels=96)

# ft2_prior = ICAPrior(tensor_names='conv2/relu:0',
#                      weighting=1e-9, name='Conv2Prior',
#                      load_path='../logs/priors/ica_prior/alexnet/conv2_relu_5x5_2000comps_1000feats/ckpt-30000',
#                      trainable=False, filter_dims=[5, 5], input_scaling=1.0, n_components=2000, n_channels=256,
#                      n_features_white=1000)

# img_prior = FoEPrior(tensor_names='pre_img/read:0',
#                      weighting=1e-3, name='ImgPrior',
#                      classifier='alexnet',
#                      filter_dims=[8, 8], input_scaling=1., n_components=512, n_channels=3,
#                      n_features_white=64*3-1)
# c2_prior = FoEPrior(tensor_names='conv2/lin:0',
#                     weighting=0, name='FoEPrior',
#                     classifier='alexnet',
#                     filter_dims=[5, 5], input_scaling=1.0, n_components=6400, n_channels=256,
#                     n_features_white=3200)

cica_img = ChannelICAPrior(tensor_names='pre_img/read:0',
                           weighting=6e-7, name='ImgCICAPrior',
                           classifier='alexnet',
                           filter_dims=[8, 8], input_scaling=1.0, n_components=200, n_channels=3,
                           n_features_white=63)


cica_c1l = ChannelICAPrior(tensor_names='conv1_lin:0',
                           weighting=1e-5, name='Conv1LinCICAPrior',
                           classifier='alexnet',
                           filter_dims=[5, 5], input_scaling=1.0, n_components=50, n_channels=96,
                           n_features_white=24)


# 2.7098e+4

modules = [split2, mse2, cica_img, split1, mse1, cica_c1l]

params = dict(classifier='alexnet',
              modules=modules,
              log_path='../logs/net_inversion/alexnet/c2_rec/mse2_cica_img_mse1_track_cica_c1l_1e-5/',
              load_path='')
params.update(mv_default_params())
params['num_iterations'] = 10000
params['learning_rate'] = 1e-10


if not os.path.exists(params['log_path']):
    os.makedirs(params['log_path'])
copyfile('./ni_tests.py', params['log_path'] + 'script.py')


ni = NetInversion(params)

# pre_img_init = np.reshape(np.load(params['log_path'] + 'mats/rec_10000.npy'), [1, 224, 224, 3])
pre_img_init = None
# 2.7098e+4
ni.train_pre_image('../data/selected/images_resized/red-fox.bmp', optim_name='adam',
                   jitter_t=0, jitter_stop_point=0, range_clip=False, scale_pre_img=1.0,
                   lr_lower_points=((0, 9e-1),),
                   save_as_plot=False, pre_img_init=pre_img_init, tensor_names_to_save=())
