from net_inversion import NetInversion
from modules.ica_prior import ICAPrior
from modules.foe_prior import FoEPrior
from modules.loss_modules import NormedMSELoss
from modules.split_module import SplitModule
from utils.parameter_utils import mv_default_params
from utils.temp_utils import make_feat_map_mats, make_reduced_feat_map_mats
from shutil import copyfile
import os

# make_feat_map_mats(num_patches=100000, map_name='conv2/lin:0', classifier='alexnet', ph=5, pw=5,
#                    save_dir='../data/patches/alexnet/conv2_lin_5x5/', whiten_mode='pca', batch_size=50)
#
# make_reduced_feat_map_mats(100000, load_dir='../data/patches/alexnet/conv2_lin_5x5/',
#                            n_features=6400, n_to_keep=3200,
#                            save_dir='../data/patches/alexnet/conv2_lin_5x5_3200feats/')

# kept 97.8% eigv fraction

# ica_prior = ICAPrior(tensor_names='conv2/relu:0',
#                      weighting=0.00001, name='ICAPrior',
#                      load_path='../logs/priors/ica_prior/alexnet/conv2_relu_5x5_10000comp_6400feats/',
#                      trainable=False, filter_dims=[5, 5], input_scaling=1.0, n_components=10000, n_channels=256,
#                      n_features_white=6399)
#
# ica_prior.train_prior(batch_size=200, num_iterations=30000,
#                       lr_lower_points=[(1, 1.0e-4)],
#                       whiten_mode='pca', data_dir='../data/patches/alexnet/conv2_relu_5x5_6399feats/',
#                       num_data_samples=100000, n_features=6399,
#                       plot_filters=False, prev_ckpt=10000)

foe_prior = FoEPrior(tensor_names='conv2/lin:0',
                     weighting=0.001, name='FoEPrior',
                     classifier='alexnet',
                     trainable=False, filter_dims=[5, 5], input_scaling=1.0, n_components=6400, n_channels=256,
                     n_features_white=3200)

if not os.path.exists(foe_prior.load_path):
    os.makedirs(foe_prior.load_path)
copyfile('./server_script.py', foe_prior.load_path + 'script.py')

foe_prior.train_prior(batch_size=500, num_iterations=80000,
                      lr_lower_points=[(0, 1e-5), (11000, 1e-6),
                                       (12000, 1e-7), (13000, 1e-8)],
                      whiten_mode='pca',
                      num_data_samples=100000,
                      plot_filters=False, prev_ckpt=10000, log_freq=500, do_clip=True)

# call 1

# split = SplitModule(name_to_split='conv4/relu:0', img_slice_name='img_rep', rec_slice_name='rec_rep')
# mse = NormedMSELoss(target='img_rep:0', reconstruction='rec_rep:0', weighting=1.)
#
# ft2_prior = ICAPrior(tensor_names='conv2/relu:0',
#                      weighting=1e-9, name='Conv2Prior',
#                      load_path='../logs/priors/ica_prior/alexnet/conv2_relu_5x5_6400comp_3200feats/ckpt-30000',
#                      trainable=False, filter_dims=[5, 5], input_scaling=1.0, n_components=6400, n_channels=256,
#                      n_features_white=3200)
#
# img_prior = ICAPrior(tensor_names='pre_img/read:0',
#                      weighting=1e-3, name='ImgPrior',
#                      load_path='../logs/priors/ica_prior/color_8x8_512comp/ckpt-10000',
#                      trainable=False, filter_dims=[8, 8], input_scaling=1.0, n_components=512, n_channels=3,
#                      n_features_white=64*3-1)
#
# modules = [split, mse, ft2_prior, img_prior]
#
# params = dict(classifier='alexnet',
#               modules=modules,
#               log_path='../logs/net_inversion/alexnet/c4_rec/mse4_pimg_pc2rich_1e-9/',
#               load_path='')
# params.update(mv_default_params())
# params['num_iterations'] = 17000
# params['learning_rate'] = 0.01
#
# if not os.path.exists(params['log_path']):
#     os.makedirs(params['log_path'])
# copyfile('./server_script.py', params['log_path'] + 'script.py')
#
# ni = NetInversion(params)
#
# ni.train_pre_image('../data/selected/images_resized/red-fox.bmp', optim_name='adam',
#                    jitter_t=0, jitter_stop_point=0, range_clip=False, scale_pre_img=2.7098e+4,
#                    lr_lower_points=((0, 1e-1), (1000, 3e-2), (1200, 1e-2), (1500, 3e-3),
#                                     (2000, 1e-3), (8000, 3e-4), (13000, 1e-4)),
#                    save_as_plot=False)
