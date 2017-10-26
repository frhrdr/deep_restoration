from net_inversion import NetInversion
from modules.inv_default_modules import default_deconv_conv_module, get_stacked_module
from shutil import copyfile
from utils.filehandling import load_image
import os
import numpy as np

classifier = 'vgg16'
module_id = 2
dc_module = default_deconv_conv_module(classifier, module_id, alt_load_subdir='solotrain')
log_path = '../logs/cnn_inversion/{}/DC{}_solo/'.format(classifier, module_id)
if not os.path.exists(log_path):
    os.makedirs(log_path)
copyfile('./stack_script_{}.py'.format(module_id), log_path + 'script.py')

# modules = module_list + [module_list[-1].get_mse_loss()]
modules = [dc_module, dc_module.get_mse_loss()]
ni = NetInversion(modules, log_path, classifier=classifier, summary_freq=100, print_freq=500, log_freq=1000)

ni.train_on_dataset(n_iterations=15000, batch_size=32, test_set_size=200, test_freq=100,
                    optim_name='adam', lr_lower_points=((0, 1e-3), (1000, 3e-4), (4000, 1e-4),
                                                        (7000, 3e-5), (10000, 1e-5), (13000, 3e-6),
                                                        (14000, 1e-6)))
