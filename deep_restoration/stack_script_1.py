from net_inversion import NetInversion
from modules.inv_default_modules import default_deconv_conv_module, get_stacked_module
from shutil import copyfile
from utils.filehandling import load_image
import os
import numpy as np

classifier = 'alexnet'
# start_layer = 4
# rec_layer = 1
# module_list = get_stacked_module(classifier=classifier, start_layer=start_layer, rec_layer=rec_layer,
#                                  alt_load_subdir='alexnet_stack_4_to_1', trainable=False)
# log_path = '../logs/cnn_inversion/{}/stack_{}_to_{}/'.format(classifier, start_layer, rec_layer)
module_id = 6
dc_module = default_deconv_conv_module(classifier, module_id, alt_load_subdir='solotrain')
log_path = '../logs/cnn_inversion/{}/DC{}_solo/'.format(classifier, module_id)
if not os.path.exists(log_path):
    os.makedirs(log_path)
copyfile('./cnn_inv_script.py', log_path + 'stack_script_8.py')


# modules = module_list + [module_list[-1].get_mse_loss()]
modules = [dc_module, dc_module.get_mse_loss()]
ni = NetInversion(modules, log_path, classifier=classifier, summary_freq=10, print_freq=10, log_freq=500)

ni.train_on_dataset(n_iterations=5000, batch_size=32, test_set_size=200, test_freq=100,
                    optim_name='adam', lr_lower_points=((0, 1e-4), (3000, 3e-5), (6000, 3e-6)))
