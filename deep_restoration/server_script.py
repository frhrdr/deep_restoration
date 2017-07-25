from net_inversion import NetInversion
from modules.ica_prior import ICAPrior
from modules.loss_modules import NormedMSELoss
from modules.split_module import SplitModule
from utils.parameter_utils import mv_default_params
from shutil import copyfile
import os

# make_feat_map_mats(num_patches=100000, map_name='conv2/relu:0', classifier='alexnet', ph=5, pw=5,
#                    save_dir='../data/patches/alexnet/conv2_relu_5x5/', whiten_mode='pca', batch_size=50)
#
# make_reduced_feat_map_mats(100000, load_dir='../data/patches/alexnet/conv2_relu_5x5/',
#                            n_features=6400, n_to_keep=3200,
#                            save_dir='../data/patches/alexnet/conv2_relu_5x5_3200feats/')

# kept 97.8% eigv fraction

# ica_prior = ICAPrior(tensor_names='conv2/relu:0',
#                      weighting=0.00001, name='ICAPrior',
#                      load_path='../logs/priors/ica_prior/alexnet/5x5_conv2_relu_6400comp_3200feats/',
#                      trainable=False, filter_dims=[5, 5], input_scaling=1.0, n_components=2000, n_channels=256,
#                      n_features_white=3200)
#
# ica_prior.train_prior(batch_size=200, num_iterations=30000,
#                       lr_lower_points=[(0, 1.0e-4), (10000, 1.0e-5), (20000, 3.0e-6), (25000, 1.0e-6)],
#                       whiten_mode='pca', data_dir='../data/patches/alexnet/conv2_relu_5x5_3200feats/',
#                       num_data_samples=100000, n_features=3200,
#                       plot_filters=False)

split = SplitModule(name_to_split='conv4/relu:0', img_slice_name='img_rep', rec_slice_name='rec_rep')
mse = NormedMSELoss(target='img_rep:0', reconstruction='rec_rep:0', weighting=1.)

ft2_prior = ICAPrior(tensor_names='conv2/relu:0',
                     weighting=1e-7, name='Conv2Prior',
                     load_path='../logs/priors/ica_prior/alexnet/5x5_conv2_relu_6400comp_3200feats/ckpt-30000',
                     trainable=False, filter_dims=[5, 5], input_scaling=1.0, n_components=6400, n_channels=256,
                     n_features_white=3200)

img_prior = ICAPrior(tensor_names='pre_img/read:0',
                     weighting=1e-3, name='ImgPrior',
                     load_path='../logs/priors/ica_prior/8by8_512_color/ckpt-10000',
                     trainable=False, filter_dims=[8, 8], input_scaling=1.0, n_components=512, n_channels=3,
                     n_features_white=64*3-1)

modules = [split, mse, ft2_prior, img_prior]

params = dict(classifier='alexnet',
              modules=modules,
              log_path='../logs/net_inversion/alexnet/c4_rec/mse4_pimg_pc2rich_1e-7/',
              load_path='')
params.update(mv_default_params())
params['num_iterations'] = 17000
params['learning_rate'] = 0.01

if not os.path.exists(params['log_path']):
    os.makedirs(params['log_path'])
copyfile('./server_script.py', params['log_path'] + 'script.py')

ni = NetInversion(params)

ni.train_pre_image('../data/selected/images_resized/red-fox.bmp', optim_name='adam',
                   jitter_t=0, jitter_stop_point=0, range_clip=False, scale_pre_img=2.7098e+4,
                   lr_lower_points=((0, 1e-1), (-1, 3e-2), (800, 1e-2), (1000, 3e-3),
                                    (1500, 1e-3), (60000, 3e-4), (9000, 1e-4)))