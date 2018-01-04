from plotting_setups import vgg_rec_collage, alex_rec_collage, vgg_mv_collage
from utils.mv_benchmark import run_mv_scripts

vgg_mv_collage('../logs/mahendran_vedaldi/2016/vgg16/image_rec/collage_rescaled.png', rescale=True)
# run_mv_scripts('vgg16', ('pool1', 'pool2', 'pool3', 'pool4', 'pool5', 'fc6/lin', 'fc7/lin', 'fc8/lin'))
# vgg_rec_collage('../logs/opt_inversion/vgg16/image_rec/collage_2123.png', rescale=False)
# vgg_rec_collage('../logs/opt_inversion/vgg16/image_rec/collage_2123_rescaled.png', rescale='perc')
# alex_rec_collage('../logs/opt_inversion/alexnet/image_rec/collage.png', rescale=False)

# from net_inversion import NetInversion
# from modules.inv_default_modules import default_deconv_conv_module, get_stacked_module
# from shutil import copyfile
# from utils.filehandling import load_image
# import os
# import numpy as np
#
# classifier = 'vgg16'
# module_id = 6
# dc_module = default_deconv_conv_module(classifier, module_id, alt_load_subdir='solotrain')
# log_path = '../logs/cnn_inversion/{}/DC{}_solo/'.format(classifier, module_id)
# if not os.path.exists(log_path):
#     os.makedirs(log_path)
# copyfile('./stack_script_{}.py'.format(module_id), log_path + 'script.py')
#
# # modules = module_list + [module_list[-1].get_mse_loss()]
# modules = [dc_module, dc_module.get_mse_loss()]
# ni = NetInversion(modules, log_path, classifier=classifier, summary_freq=100, print_freq=500, log_freq=1000)
#
# ni.train_on_dataset(n_iterations=5000, batch_size=32, test_set_size=200, test_freq=100,
#                     optim_name='adam', lr_lower_points=((0, 3e-5),))
