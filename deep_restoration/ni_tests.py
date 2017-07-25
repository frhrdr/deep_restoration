from net_inversion import NetInversion
from modules.ica_prior import ICAPrior
from modules.loss_modules import NormedMSELoss, SoftRangeLoss, TotalVariationLoss
from modules.split_module import SplitModule
from modules.norm_module import NormModule
from utils.parameter_utils import mv_default_params
from shutil import copyfile
import os

split = SplitModule(name_to_split='conv4/relu:0', img_slice_name='img_rep', rec_slice_name='rec_rep')
mse_weight = 1.
mse = NormedMSELoss(target='img_rep:0', reconstruction='rec_rep:0', weighting=mse_weight)

# ica_prior = ICAPrior(tensor_names='pool1:0',
#                      weighting=0.001, name='ICAPrior',
#                      load_path='../logs/priors/ica_prior/alexnet/3by3_pool1/ckpt-20000',
#                      trainable=False, filter_dims=[3, 3], input_scaling=1.0, n_components=512, n_channels=96)

# ft2_prior = ICAPrior(tensor_names='conv2/relu:0',
#                      weighting=1e-9, name='Conv2Prior',
#                      load_path='../logs/priors/ica_prior/alexnet/5x5_conv2_relu_2kcomp_1kfeats/ckpt-30000',
#                      trainable=False, filter_dims=[5, 5], input_scaling=1.0, n_components=2000, n_channels=256,
#                      n_features_white=1000)

img_prior = ICAPrior(tensor_names='pre_img/read:0',
                     weighting=1e-3, name='ImgPrior',
                     load_path='../logs/priors/ica_prior/8by8_512_color/ckpt-10000',
                     trainable=False, filter_dims=[8, 8], input_scaling=1/2.7098e+4, n_components=512, n_channels=3,
                     n_features_white=64*3-1)
# 2.7098e+4

modules = [split, mse, img_prior]

params = dict(classifier='alexnet',
              modules=modules,
              log_path='../logs/net_inversion/alexnet/c4_rec/mse4_pimg_noscale/',
              load_path='')
params.update(mv_default_params())
params['num_iterations'] = 17000
params['learning_rate'] = 0.01

if not os.path.exists(params['log_path']):
    os.makedirs(params['log_path'])
copyfile('./ni_tests.py', params['log_path'] + 'script.py')


ni = NetInversion(params)

ni.train_pre_image('../data/selected/images_resized/red-fox.bmp', optim_name='adam',
                   jitter_t=0, jitter_stop_point=0, range_clip=False, scale_pre_img=1.0,
                   lr_lower_points=((0, 1e-1), (-1, 3e-2), (800, 1e-2), (1000, 3e-3),
                                    (1500, 1e-3), (60000, 3e-4), (9000, 1e-4)))
