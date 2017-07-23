from net_inversion import NetInversion
from modules.ica_prior import ICAPrior
from modules.loss_modules import NormedMSELoss, SoftRangeLoss, TotalVariationLoss
from modules.split_module import SplitModule
from modules.norm_module import NormModule
from utils.parameter_utils import mv_default_params

imagenet_mean = (123.68 + 116.779 + 103.939) / 3
imagenet_mean = [123.68, 116.779, 103.939]
split = SplitModule(name_to_split='conv4/relu:0', img_slice_name='img_rep', rec_slice_name='rec_rep')
# norm = NormModule(name_to_norm='pre_img/read:0', out_name='pre_img_normed', offset=imagenet_mean, scale=1.0)
mse_weight = 1.
mse = NormedMSELoss(target='img_rep:0', reconstruction='rec_rep:0', weighting=mse_weight)

# ica_prior = ICAPrior(tensor_names='pool1:0',
#                      weighting=0.001, name='ICAPrior',
#                      load_path='../logs/priors/ica_prior/alexnet/3by3_pool1/ckpt-20000',
#                      trainable=False, filter_dims=[3, 3], input_scaling=1.0, n_components=512, n_channels=96)

ft2_prior = ICAPrior(tensor_names='conv2/relu:0',
                     weighting=0.0001, name='Conv2Prior',
                     load_path='../logs/priors/ica_prior/alexnet/5x5_conv2_relu_2kcomp_1kfeats/ckpt-30000',
                     trainable=False, filter_dims=[5, 5], input_scaling=1.0, n_components=2000, n_channels=256,
                     n_features_white=1000)

img_prior = ICAPrior(tensor_names='pre_img/read:0',
                     weighting=0.001, name='ImgPrior',
                     load_path='../logs/priors/ica_prior/8by8_512_color/ckpt-10000',
                     trainable=False, filter_dims=[8, 8], input_scaling=1.0, n_components=512, n_channels=3,
                     n_features_white=64*3-1)


modules = [split, mse, ft2_prior, img_prior]

params = dict(classifier='alexnet',
              modules=modules,
              log_path='../logs/net_inversion/alexnet/c4_rec/mse4_p2_pimg/',
              load_path='')
params.update(mv_default_params())
params['num_iterations'] = 10000
params['learning_rate'] = 0.01

ni = NetInversion(params)

# ni.train_pre_image('../data/selected/images_resized/val13_monkey.bmp', jitter_t=0, optim_name='adam',
#                    lr_lower_points=((0, 0.01), (500, 0.001), (1000, 0.0001), (3000, 0.00003), (7000, 0.00001)))
ni.train_pre_image('../data/selected/images_resized/red-fox.bmp', optim_name='adam',
                   jitter_t=0, jitter_stop_point=0, range_clip=False, scale_pre_img=2.7098e+4,
                   lr_lower_points=((0, 0.01), (1000, 0.003), (1200, 0.001), (3000, 0.0003),
                                    (6000, 0.0001), (6000, 0.00003), (9000, 0.00001)))
