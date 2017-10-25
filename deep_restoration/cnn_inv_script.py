from net_inversion import NetInversion
from modules.inv_modules import ScaleConvConvModule, DeconvConvModule
from modules.loss_modules import MSELoss
from shutil import copyfile
from utils.filehandling import load_image
import os
import numpy as np

dc4 = DeconvConvModule(inv_input_name='conv2/lin:0', inv_target_name='pool1:0',
                       hidden_channels=256, rec_name='pool1_rec',
                       op1_hw=[8, 8], op1_strides=[1, 1, 1, 1], op2_hw=[8, 8], op2_strides=[1, 1, 1, 1],
                       name='DC4',
                       subdir='solotrain', trainable=True)

dc3 = DeconvConvModule(inv_input_name='pool1:0', inv_target_name='lrn1:0',
                       hidden_channels=96, rec_name='lrn1_rec',
                       op1_hw=[3, 3], op1_strides=[1, 2, 2, 1], op2_hw=[8, 8], op2_strides=[1, 1, 1, 1],
                       op1_pad='VALID', op2_pad='SAME',
                       name='DC3',
                       subdir='solotrain', trainable=True)

dc2 = DeconvConvModule(inv_input_name='lrn1:0', inv_target_name='conv1/lin:0',
                       hidden_channels=96, rec_name='c1l_rec',
                       op1_hw=[8, 8], op1_strides=[1, 1, 1, 1], op2_hw=[8, 8], op2_strides=[1, 1, 1, 1],
                       name='DC2',
                       subdir='solotrain', trainable=True)

dc1 = DeconvConvModule(inv_input_name='conv1/lin:0', inv_target_name='rgb_scaled:0',
                       hidden_channels=96, rec_name='rgb_rec',
                       op1_hw=[11, 11], op1_strides=[1, 4, 4, 1], op2_hw=[11, 11], op2_strides=[1, 1, 1, 1],
                       op1_pad='VALID',
                       name='DC1',
                       subdir='solotrain', trainable=True)

mse4 = MSELoss(target='pool1:0', reconstruction='DC4/pool1_rec:0', name='MSE_pool1')
mse3 = MSELoss(target='lrn1:0', reconstruction='DC3/lrn1_rec:0', name='MSE_lrn1')
mse2 = MSELoss(target='conv1/lin:0', reconstruction='DC2/c1l_rec:0', name='MSE_c1l')
mse1 = MSELoss(target='rgb_scaled:0', reconstruction='DC1/rgb_rec:0', name='MSE_rgb')

log_path = '../logs/cnn_inversion/alexnet/DC4_solo/'
if not os.path.exists(log_path):
    os.makedirs(log_path)
copyfile('./cnn_inv_script.py', log_path + 'script.py')

# modules = [dc1, dc2, dc3, mse1, mse2, mse3, dc4, mse4]
modules = [dc4, mse4]
ni = NetInversion(modules, log_path, classifier='alexnet', summary_freq=10, print_freq=10, log_freq=500)

ni.train_on_dataset(n_iterations=3000, batch_size=32, test_set_size=200, test_freq=100,
                    optim_name='adam', lr_lower_points=((0, 3e-5), (1000, 1e-5), (2000, 3e-6)))

# dc2.trainable = False
# image_file = '../data/selected/images_resized_227/red-fox.bmp'
# img_mat = np.expand_dims(load_image(image_file), axis=0)
#
# to_fetch = ('DC1/rgb_rec:0',)
# rec = ni.run_model_on_images(img_mat, to_fetch)[0]
# print(rec.shape)
# np.save('cnn_rec.npy', rec)
