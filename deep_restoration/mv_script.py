from net_inversion import NetInversion
import numpy as np
from modules.loss_modules import NormedMSELoss, SoftRangeLoss, TotalVariationLoss
from modules.split_module import SplitModule
from modules.norm_module import NormModule

imagenet_mean = (123.68 + 116.779 + 103.939) / 3
# imagenet_mean = [123.68, 116.779, 103.939]
split = SplitModule(name_to_split='conv4/relu:0', img_slice_name='img_rep', rec_slice_name='rec_rep')
norm = NormModule(name_to_norm='pre_img/read:0', out_name='pre_img_normed', offset=imagenet_mean, scale=1.0)
mse_weight = 100.
mse = NormedMSELoss(target='img_rep:0', reconstruction='rec_rep:0', weighting=mse_weight)

sr_weight = 1. / (224. * 224. * 80. ** 6.)
sr_prior = SoftRangeLoss(tensor='pre_img_normed:0', alpha=6, weighting=sr_weight)
tv_weight = 1. / (224. * 224. * (80. / 6.5) ** 2.)
tv_prior = TotalVariationLoss(tensor='pre_img_normed:0', beta=2, weighting=tv_weight)


modules = [split, norm, mse, sr_prior, tv_prior]

classifier = 'alexnet'

log_path = '../logs/mahendran_vedaldi/2016/repro/alexnet/conv4_relu/fox/'
img_path = '../data/selected/images_resized/red-fox.bmp'

load_path = ''
optimizer = 'momentum'
print_freq = 50
log_freq = 500
summary_freq = 10
n_iterations = 12000
lr_factor = 80. ** 2 / 6.
lr_lower_points = ((0, 1e-2 * lr_factor), (1500, 3e-3 * lr_factor), (3000, 1e-3 * lr_factor))

ni = NetInversion(modules, log_path, classifier='alexnet')
pre_img_init = np.reshape(np.load(log_path + 'mats/rec_500.npy'), [1, 224, 224, 3])
# pre_img_init = np.load('../logs/net_inversion/alexnet/c1l_tests_16_08/init_helper.npy')
# pre_img_init = None

# ni.train_pre_image('../data/selected/images_resized/val13_monkey.bmp', jitter_t=0, optim_name='adam',
#                    lr_lower_points=((0, 0.01), (500, 0.001), (1000, 0.0001), (3000, 0.00003), (7000, 0.00001)))
ni.train_pre_featmap(img_path, n_iterations=n_iterations, optim_name='adam',
                     jitter_t=4, jitter_stop_point=3000, range_clip=True, scale_pre_img=1., range_b=80,
                     lr_lower_points=lr_lower_points,
                     pre_featmap_init=pre_img_init, ckpt_offset=0, save_as_plot=True)
