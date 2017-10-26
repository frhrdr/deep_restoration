from net_inversion import NetInversion
import numpy as np
from modules.loss_modules import NormedMSELoss, SoftRangeLoss, TotalVariationLoss
from modules.split_module import SplitModule
from modules.norm_module import NormModule
from utils.filehandling import load_image
from shutil import copyfile
import os
classifier = 'alexnet'

src_layer = 'fc6/lin:0'
log_path = '../logs/mahendran_vedaldi/2016/alexnet/{}/fox/'.format(src_layer.replace('/', '_')[:-len(':0')])
img_path = '../data/selected/images_resized_227/red-fox.bmp'
load_path = ''

if not os.path.exists(log_path):
    os.makedirs(log_path)
copyfile('./mv_script.py', log_path + 'mv_script.py')

mse_weight = 1.  # 300 up to c3r, 100 up to c4r, 20 up to c5r, then 1
n_iterations = 3500
range_b = 80
alpha = 6
beta = 2
jitter_t = 8  # 1 up to lrn1, 2 up to lrn2, 4 up to c5r, then 8
jitter_stop_point = 2750

if classifier == 'alexnet':
    imagenet_mean = (123.68 + 116.779 + 103.939) / 3
    img_hw = 227
else:
    imagenet_mean = [123.68, 116.779, 103.939]
    img_hw = 224

sr_weight = 1. / (img_hw ** 2 * range_b ** alpha)
tv_weight = 1. / (img_hw ** 2 * (range_b / 6.5) ** beta)

split = SplitModule(name_to_split=src_layer, img_slice_name='img_rep', rec_slice_name='rec_rep')
norm = NormModule(name_to_norm='pre_featmap/read:0', out_name='pre_featmap_normed', offset=imagenet_mean, scale=1.0)
mse = NormedMSELoss(target='img_rep:0', reconstruction='rec_rep:0', weighting=mse_weight)
sr_prior = SoftRangeLoss(tensor='pre_featmap_normed:0', alpha=6, weighting=sr_weight)
tv_prior = TotalVariationLoss(tensor='pre_featmap_normed:0', beta=2, weighting=tv_weight)

modules = [split, norm, mse, sr_prior, tv_prior]

lr_factor = 80. ** 2 / 6.
lr_lower_points = ((0, 1e-2 * lr_factor), (1000, 3e-3 * lr_factor), (2000, 1e-3 * lr_factor))

ni = NetInversion(modules, log_path, classifier='alexnet')
pre_img_init = np.expand_dims(load_image(img_path), axis=0).astype(np.float32)

# pre_img_init = np.load('../logs/net_inversion/alexnet/c1l_tests_16_08/init_helper.npy')
# pre_img_init = None

# ni.train_pre_image('../data/selected/images_resized/val13_monkey.bmp', jitter_t=0, optim_name='adam',
#                    lr_lower_points=((0, 0.01), (500, 0.001), (1000, 0.0001), (3000, 0.00003), (7000, 0.00001)))

ni.train_pre_featmap(img_path, n_iterations=n_iterations, optim_name='adam',
                     jitter_t=jitter_t, jitter_stop_point=jitter_stop_point, range_clip=True, scale_pre_img=1.,
                     range_b=range_b,
                     lr_lower_points=lr_lower_points,
                     pre_featmap_init=pre_img_init, ckpt_offset=0, save_as_plot=True)
