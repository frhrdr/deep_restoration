from net_inversion import NetInversion
from modules.inv_default_modules import get_stacked_module
from shutil import copyfile
import os

classifier = 'alexnet'
start_layer = 4
rec_layer = 2
module_list = get_stacked_module(classifier=classifier, alt_load_subdir='alexnet_stack_{}_to_{}'.format(start_layer,
                                                                                                        rec_layer),
                                 start_layer=start_layer, rec_layer=rec_layer)
log_path = '../logs/cnn_inversion/{}/stack_{}_to_{}/'.format(classifier, start_layer, rec_layer)

if not os.path.exists(log_path):
    os.makedirs(log_path)
copyfile('./stack_script_1.py', log_path + 'script.py')

modules = module_list + [module_list[-1].get_mse_loss()]
ni = NetInversion(modules, log_path, classifier=classifier, summary_freq=10, print_freq=10, log_freq=500)

ni.train_on_dataset(n_iterations=3000, batch_size=32, test_set_size=200, test_freq=100,
                    optim_name='adam', lr_lower_points=((0, 1e-5), (1000, 1e-5), (2000, 3e-6)))
