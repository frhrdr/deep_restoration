from net_inversion import NetInversion
from modules.inv_modules import ScaleConvConvModule, DeconvConvModule
from modules.loss_modules import MSELoss
from shutil import copyfile
import os


log_path = '../logs/cnn_inversion/alexnet/c2l_to_c1l_3dc'
if not os.path.exists(log_path):
    os.makedirs(log_path)
copyfile('./cnn_inv_script.py', log_path + 'script.py')

dc1 = DeconvConvModule(inv_input_name='conv2/lin:0', inv_target_name='pool1:0',
                       hidden_channels=256, rec_name='pool1_rec',
                       op1_hw=[11, 11], op1_strides=[1, 1, 1, 1], op2_hw=[11, 11], op2_strides=[1, 1, 1, 1],
                       name='DC1',
                       subdir='run1', trainable=True)

dc2 = DeconvConvModule(inv_input_name='DC1/pool1_rec:0', inv_target_name='lrn1:0',
                       hidden_channels=96, rec_name='lrn1_rec',
                       op1_hw=[11, 11], op1_strides=[1, 2, 2, 1], op2_hw=[11, 11], op2_strides=[1, 1, 1, 1],
                       name='DC2',
                       subdir='run1', trainable=True)


dc3 = DeconvConvModule(inv_input_name='DC2/lrn1_rec:0', inv_target_name='conv1/lin:0',
                       hidden_channels=96, rec_name='c1l_rec',
                       op1_hw=[11, 11], op1_strides=[1, 1, 1, 1], op2_hw=[11, 11], op2_strides=[1, 1, 1, 1],
                       name='DC3',
                       subdir='run1', trainable=True)


mse1 = MSELoss(target='pool1:0', reconstruction='DC1/pool1_rec:0', name='MSE_pool1')
mse2 = MSELoss(target='lrn1:0', reconstruction='DC2/lrn1_rec:0', name='MSE_lrn1')
mse3 = MSELoss(target='conv1/lin:0', reconstruction='DC3/c1l_rec:0', name='MSE_c1l')

modules = [dc1, dc2, dc3, mse1, mse2, mse3]

ni = NetInversion(modules, log_path, classifier='alexnet', summary_freq=10, print_freq=50, log_freq=500)

ni.train_on_dataset(n_iterations=500, batch_size=6, test_set_size=200, test_freq=100,
                    optim_name='adam', lr_lower_points=((0, 3e-4),))

# dc1.trainable = False
# image_file = '../data/selected/images_resized/red-fox.bmp'
# to_fetch = ('C1R_to_BGR_DCModule/bgr_normed_rec:0',)
# rec = ni.run_model_on_image(image_file, to_fetch)[0]
# print(rec.shape)
