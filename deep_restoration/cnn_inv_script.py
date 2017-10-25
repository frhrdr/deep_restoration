from net_inversion import NetInversion
from modules.inv_modules import ScaleConvConvModule, DeconvConvModule
from modules.loss_modules import MSELoss
from modules.inv_default_modules import default_deconv_conv_module, get_stacked_module
from shutil import copyfile
from utils.filehandling import load_image
import os
import numpy as np

classifier = 'alexnet'
start_layer = 4
rec_layer = 1
module_list = get_stacked_module(classifier=classifier, start_layer=start_layer, rec_layer=rec_layer)
log_path = '../logs/cnn_inversion/{}/stack_{}_to_{}/'.format(classifier, start_layer, rec_layer)

if not os.path.exists(log_path):
    os.makedirs(log_path)
copyfile('./cnn_inv_script.py', log_path + 'script.py')


modules = module_list + [module_list[-1].get_mse_loss()]
ni = NetInversion(modules, log_path, classifier=classifier, summary_freq=10, print_freq=10, log_freq=500)

ni.train_on_dataset(n_iterations=3000, batch_size=32, test_set_size=200, test_freq=100,
                    optim_name='adam', lr_lower_points=((0, 3e-4), (1000, 1e-4), (2000, 3e-5)))

# dc2.trainable = False
# image_file = '../data/selected/images_resized_227/red-fox.bmp'
# img_mat = np.expand_dims(load_image(image_file), axis=0)
#
# to_fetch = ('DC1/rgb_rec:0',)
# rec = ni.run_model_on_images(img_mat, to_fetch)[0]
# print(rec.shape)
# np.save('cnn_rec.npy', rec)
