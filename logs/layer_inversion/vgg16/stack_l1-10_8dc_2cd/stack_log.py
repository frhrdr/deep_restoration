from parameter_utils import default_params, selected_images
from deep_restoration.old.layer_inversion import run_stacked_models

spec1 = dict(op1_height=5, op1_width=5, op1_strides=[1, 2, 2, 1],
             op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
             hidden_channels=256, target_shape=[28, 28, 256])

param1 = dict(classifier='vgg16', inv_input_name='pool3:0', inv_target_name='conv3_3/relu:0',
              inv_model_type='deconv_conv',
              inv_model_specs=spec1,
              log_path='./logs/layer_inversion/vgg16/l10_dc/run1/',
              load_path='./logs/layer_inversion/vgg16/l10_dc/run1/ckpt-5000')


spec2 = dict(op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
             op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
             hidden_channels=256, target_shape=[56, 56, 256])

param2 = dict(classifier='vgg16', inv_input_name='conv3_3/relu:0', inv_target_name='conv3_2/relu:0',
              inv_model_type='deconv_conv',
              inv_model_specs=spec2,
              log_path='./logs/layer_inversion/vgg16/l9_dc/run1/',
              load_path='./logs/layer_inversion/vgg16/l9_dc/run1/ckpt-5000')


spec3 = dict(op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
             op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
             hidden_channels=256, target_shape=[56, 56, 256])

param3 = dict(classifier='vgg16', inv_input_name='conv3_2/relu:0', inv_target_name='conv3_1/relu:0',
              inv_model_type='deconv_conv',
              inv_model_specs=spec3,
              log_path='./logs/layer_inversion/vgg16/l8_dc/run1/',
              load_path='./logs/layer_inversion/vgg16/l8_dc/run1/ckpt-5000')


spec4 = dict(op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
             op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
             hidden_channels=256, target_shape=[56, 56, 128])

param4 = dict(classifier='vgg16', inv_input_name='conv3_1/relu:0', inv_target_name='pool2:0',
              inv_model_type='deconv_conv',
              inv_model_specs=spec4,
              log_path='./logs/layer_inversion/vgg16/l7_dc/run1/',
              load_path='./logs/layer_inversion/vgg16/l7_dc/run1/ckpt-5000')


spec5 = dict(op1_height=5, op1_width=5, op1_strides=[1, 2, 2, 1],
             op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
             hidden_channels=128, target_shape=[112, 112, 128])

param5 = dict(classifier='vgg16', inv_input_name='pool2:0', inv_target_name='conv2_2/relu:0',
              inv_model_type='deconv_conv',
              inv_model_specs=spec5,
              log_path='./logs/layer_inversion/vgg16/l6_dc/run1/',
              load_path='./logs/layer_inversion/vgg16/l6_dc/run1/ckpt-3000')


spec6 = dict(op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
             op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
             hidden_channels=128, target_shape=[112, 112, 128])

param6 = dict(classifier='vgg16', inv_input_name='conv2_2/relu:0', inv_target_name='conv2_1/relu:0',
              inv_model_type='deconv_conv',
              inv_model_specs=spec6,
              log_path='./logs/layer_inversion/vgg16/l5_dc/run1/',
              load_path='./logs/layer_inversion/vgg16/l5_dc/run1/ckpt-3000')


spec7 = dict(op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
             op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
             hidden_channels=128, target_shape=[112, 112, 128])

param7 = dict(classifier='vgg16', inv_input_name='conv2_1/relu:0', inv_target_name='pool1:0',
              inv_model_type='deconv_conv',
              inv_model_specs=spec7,
              log_path='./logs/layer_inversion/vgg16/l4_dc/run1/',
              load_path='./logs/layer_inversion/vgg16/l4_dc/run1/ckpt-3000')

spec8 = dict(op1_height=5, op1_width=5, op1_strides=[1, 2, 2, 1],
             op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
             hidden_channels=64)
param8 = dict(classifier='vgg16', inv_input_name='pool1:0', inv_target_name='conv1_2/relu:0',
              inv_model_type='deconv_conv',
              inv_model_specs=spec8,
              log_path='./logs/layer_inversion/vgg16/l3_dc/run1/',
              load_path='./logs/layer_inversion/vgg16/l3_dc/run1/ckpt-3000')

spec9 = dict(op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
             op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
             hidden_channels=64)
param9 = dict(classifier='vgg16', inv_input_name='conv1_2/relu:0', inv_target_name='conv1_1/relu:0',
              inv_model_type='conv_deconv',
              inv_model_specs=spec9,
              log_path='./logs/layer_inversion/vgg16/l2_cd/run1/',
              load_path='./logs/layer_inversion/vgg16/l2_cd/run1/ckpt-3000')

spec10 = dict(op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
              hidden_channels=64)
param10 = dict(classifier='vgg16', inv_input_name='conv1_1/relu:0', inv_target_name='bgr_normed:0',
               inv_model_type='conv_deconv',
               inv_model_specs=spec10,
               log_path='./logs/layer_inversion/vgg16/l1_cd/run3/',
               load_path='./logs/layer_inversion/vgg16/l1_cd/run3/ckpt-3000')


param1.update(default_params())
param2.update(default_params())
param3.update(default_params())
param4.update(default_params())
param5.update(default_params())
param6.update(default_params())
param7.update(default_params())
param8.update(default_params())
param9.update(default_params())
param10.update(default_params())

run_stacked_models([param1, param2, param3, param4, param5, param6, param7, param8, param9, param10])
run_stacked_models([selected_images(param1), selected_images(param2),
                    selected_images(param3), selected_images(param4),
                    selected_images(param5), selected_images(param6),
                    selected_images(param7), selected_images(param8),
                    selected_images(param9), selected_images(param10)], file_name='stacked_selected')

