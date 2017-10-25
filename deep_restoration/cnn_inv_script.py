from net_inversion import NetInversion
from modules.inv_modules import ScaleConvConvModule, DeconvConvModule
from modules.loss_modules import MSELoss
from shutil import copyfile
import os
import numpy as np

log_path = '../logs/cnn_inversion/alexnet/c2l_to_c1l_3dc'
if not os.path.exists(log_path):
    os.makedirs(log_path)
copyfile('./cnn_inv_script.py', log_path + 'script.py')

dc1 = DeconvConvModule(inv_input_name='conv2/lin:0', inv_target_name='pool1:0',
                       hidden_channels=256, rec_name='pool1_rec',
                       op1_hw=[8, 8], op1_strides=[1, 1, 1, 1], op2_hw=[8, 8], op2_strides=[1, 1, 1, 1],
                       name='DC1',
                       subdir='run1', trainable=True)

dc2 = DeconvConvModule(inv_input_name='DC1/pool1_rec:0', inv_target_name='lrn1:0',
                       hidden_channels=96, rec_name='lrn1_rec',
                       op1_hw=[3, 3], op1_strides=[1, 2, 2, 1], op2_hw=[8, 8], op2_strides=[1, 1, 1, 1],
                       op1_pad='VALID', op2_pad='SAME',
                       name='DC2',
                       subdir='run1', trainable=True)


dc3 = DeconvConvModule(inv_input_name='DC2/lrn1_rec:0', inv_target_name='conv1/lin:0',
                       hidden_channels=96, rec_name='c1l_rec',
                       op1_hw=[8, 8], op1_strides=[1, 1, 1, 1], op2_hw=[8, 8], op2_strides=[1, 1, 1, 1],
                       name='DC3',
                       subdir='run1', trainable=True)

dc4 = DeconvConvModule(inv_input_name='DC3/c1l_rec:0', inv_target_name='rgb_scaled:0',
                       hidden_channels=96, rec_name='rgb_rec',
                       op1_hw=[11, 11], op1_strides=[1, 4, 4, 1], op2_hw=[11, 11], op2_strides=[1, 1, 1, 1],
                       op1_pad='VALID',
                       name='DC4',
                       subdir='run1', trainable=True)

mse1 = MSELoss(target='pool1:0', reconstruction='DC1/pool1_rec:0', name='MSE_pool1')
mse2 = MSELoss(target='lrn1:0', reconstruction='DC2/lrn1_rec:0', name='MSE_lrn1')
mse3 = MSELoss(target='conv1/lin:0', reconstruction='DC3/c1l_rec:0', name='MSE_c1l')
mse4 = MSELoss(target='rgb_scaled:0', reconstruction='DC4/rgb_rec:0', name='MSE_rgb')

modules = [dc1, dc2, dc3, mse1, mse2, mse3, dc4, mse4]
ni = NetInversion(modules, log_path, classifier='alexnet', summary_freq=10, print_freq=10, log_freq=500)

# ni.train_on_dataset(n_iterations=5000, batch_size=20, test_set_size=200, test_freq=100,
#                     optim_name='adam', lr_lower_points=((0, 3e-3), (2000, 3e-4),
#                                                         (4000, 3e-5),))

dc1.trainable = False
image_file = '../data/selected/images_resized_227/red-fox.bmp'
to_fetch = ('DC4/rgb_rec:0',)
rec = ni.run_model_on_image(image_file, to_fetch)[0]
print(rec.shape)
np.save('ccn_rec.npy', rec)
