import numpy as np
import tensorflow as tf
from net_inversion import NetInversion
from modules.inv_modules import ScaleConvConvModule, DeconvConvModule

scc1 = DeconvConvModule()

modules = []
log_path='../logs/cnn_inversion/alexnet/pre_featmap/88/1e-10/'

ni = NetInversion(modules, log_path, classifier='alexnet', summary_freq=10, print_freq=50, log_freq=500)

ni.train_on_dataset(n_iterations=100, batch_size=10, test_set_size=200, test_freq=100,
                    optim_name='adam', lr_lower_points=((0, 1e-4),))

# spec1 = dict(inv_model_type='deconv_conv',
#              op1_height=5, op1_width=5, op1_strides=[1, 2, 2, 1], op1_pad='SAME',
#              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1], op2_pad='SAME',
#              hidden_channels=256, target_shape=[28, 28, 256],
#              inv_input_name='pool3:0', inv_target_name='conv3_3/relu:0',
#              rec_name='conv3_3_rec', add_loss=True)
#
# spec2 = dict(inv_model_type='deconv_conv',
#              op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1], op1_pad='SAME',
#              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1], op2_pad='SAME',
#              hidden_channels=256, target_shape=[56, 56, 256],
#              inv_input_name='module_1/conv3_3_rec:0', inv_target_name='conv3_2/relu:0',
#              rec_name='conv3_2_rec', add_loss=True)
#
# spec3 = dict(inv_model_type='deconv_conv',
#              op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1], op1_pad='SAME',
#              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1], op2_pad='SAME',
#              hidden_channels=256, target_shape=[56, 56, 256],
#              inv_input_name='module_2/conv3_2_rec:0', inv_target_name='conv3_1/relu:0',
#              rec_name='conv3_1_rec', add_loss=True)
#
# spec4 = dict(inv_model_type='deconv_conv',
#              op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1], op1_pad='SAME',
#              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1], op2_pad='SAME',
#              hidden_channels=256, target_shape=[56, 56, 128],
#              inv_input_name='module_3/conv3_1_rec:0', inv_target_name='pool2:0',
#              rec_name='pool2_rec', add_loss=True)
#
# spec5 = dict(inv_model_type='deconv_conv',
#              op1_height=5, op1_width=5, op1_strides=[1, 2, 2, 1], op1_pad='SAME',
#              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1], op2_pad='SAME',
#              hidden_channels=128, target_shape=[112, 112, 128],
#              inv_input_name='module_4/pool2_rec:0', inv_target_name='conv2_2/relu:0',
#              rec_name='conv2_2_rec', add_loss=True)
#
# spec6 = dict(inv_model_type='deconv_conv',
#              op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1], op1_pad='SAME',
#              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1], op2_pad='SAME',
#              hidden_channels=128, target_shape=[112, 112, 128],
#              inv_input_name='module_5/conv2_2_rec:0', inv_target_name='conv2_1/relu:0',
#              rec_name='conv2_1_rec', add_loss=True)
#
# spec7 = dict(inv_model_type='deconv_conv',
#              op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1], op1_pad='SAME',
#              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1], op2_pad='SAME',
#              hidden_channels=128, target_shape=[112, 112, 64],
#              inv_input_name='module_6/conv2_1_rec:0', inv_target_name='pool1:0',
#              rec_name='pool1_rec', add_loss=True)
#
# spec8 = dict(inv_model_type='deconv_conv',
#              op1_height=5, op1_width=5, op1_strides=[1, 2, 2, 1], op1_pad='SAME',
#              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1], op2_pad='SAME',
#              hidden_channels=64, target_shape=[224, 224, 64],
#              inv_input_name='module_7/pool1_rec:0', inv_target_name='conv1_2/relu:0',
#              rec_name='conv1_2_rec', add_loss=True)
#
# spec9 = dict(inv_model_type='deconv_conv',
#              op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1], op1_pad='SAME',
#              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1], op2_pad='SAME',
#              hidden_channels=64, target_shape=[224, 224, 64],
#              inv_input_name='module_8/conv1_2_rec:0', inv_target_name='conv1_1/relu:0',
#              rec_name='conv1_1_rec', add_loss=True)
#
# spec10 = dict(inv_model_type='deconv_conv',
#               op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1], op1_pad='SAME',
#               op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1], op2_pad='SAME',
#               hidden_channels=64, target_shape=[224, 224, 3],
#               inv_input_name='module_9/conv1_1_rec:0', inv_target_name='bgr_normed:0',
#               rec_name='reconstruction', add_loss=True)
#
# params = dict(classifier='vgg16',
#               inv_model_specs=[spec1, spec2, spec3, spec4, spec5, spec6, spec7, spec8, spec9, spec10],
#               log_path='./logs/net_inversion/vgg16/l1-10_10dc/run6/',
#               load_path='')
# params.update(default_params())
# params['learning_rate'] = 3.0e-5
# print(params)
# ni = NetInversion(params)
# ni.visualize()