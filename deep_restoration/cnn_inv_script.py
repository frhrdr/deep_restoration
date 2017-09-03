import numpy as np
import tensorflow as tf
from net_inversion import NetInversion
from modules.inv_modules import ScaleConvConvModule, DeconvConvModule
from modules.loss_modules import NormedMSELoss

scc1 = DeconvConvModule(inv_input_name='conv1/relu:0', inv_target_name='bgr_normed:0',
                        hidden_channels=96, rec_name='bgr_normed_rec',
                        op1_hw=[11, 11], op1_strides=[1, 4, 4, 1], op2_hw=[11, 11], op2_strides=[1, 1, 1, 1],
                        name='ScaleConvConvModule',
                        subdir='run1', trainable=False)

nmse = NormedMSELoss(target='bgr_normed:0', reconstruction='ScaleConvConvModule/bgr_normed_rec:0', name='NMSE')

modules = [scc1, nmse]
log_path='../logs/cnn_inversion/alexnet/pre_featmap/88/1e-10/'

ni = NetInversion(modules, log_path, classifier='alexnet', summary_freq=10, print_freq=50, log_freq=500)

ni.train_on_dataset(n_iterations=500, batch_size=6, test_set_size=200, test_freq=100,
                    optim_name='adam', lr_lower_points=((0, 3e-4),))

# image_file = '../data/selected/images_resized/red-fox.bmp'
# to_fetch = ('ScaleConvConvModule/bgr_normed_rec:0',)
# rec = ni.run_model_on_image(image_file, to_fetch)[0]
# print(rec.shape)