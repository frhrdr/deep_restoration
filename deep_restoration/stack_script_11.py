from net_inversion import NetInversion
from modules.inv_default_modules import get_stacked_module
from shutil import copyfile
import os

classifier = 'vgg16'
start_layer = 10
rec_layer = 1
module_list = get_stacked_module(classifier=classifier, start_layer=start_layer, rec_layer=rec_layer)
log_path = '../logs/cnn_inversion/{}/stack_{}_to_{}/'.format(classifier, start_layer, rec_layer)

if not os.path.exists(log_path):
    os.makedirs(log_path)
copyfile('./cnn_inv_script.py', log_path + 'stack_script_6.py')


modules = module_list + [module_list[-1].get_mse_loss()]
ni = NetInversion(modules, log_path, classifier=classifier, summary_freq=10, print_freq=10, log_freq=500)

ni.train_on_dataset(n_iterations=3000, batch_size=32, test_set_size=200, test_freq=100,
                    optim_name='adam', lr_lower_points=((0, 3e-5), (1000, 1e-5), (2000, 3e-6)))
