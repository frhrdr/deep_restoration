from layer_inversion import run_stacked_models
from parameter_utils import default_params, selected_images


spec1 = dict(op1_height=5, op1_width=5, op1_strides=[1, 2, 2, 1],
             op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
             hidden_channels=128, target_shape=[112, 112, 128])

param1 = dict(classifier='vgg16', inv_input_name='pool2:0', inv_target_name='conv2_2/relu:0',
              inv_model_type='deconv_conv',
              inv_model_specs=spec1,
              log_path='./logs/layer_inversion/vgg16/l6_dc/run1/',
              load_path='./logs/layer_inversion/vgg16/l6_dc/run1/ckpt-3000')


spec2 = dict(op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
             op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
             hidden_channels=128, target_shape=[112, 112, 128])

param2 = dict(classifier='vgg16', inv_input_name='conv2_2/relu:0', inv_target_name='conv2_1/relu:0',
              inv_model_type='deconv_conv',
              inv_model_specs=spec2,
              log_path='./logs/layer_inversion/vgg16/l5_dc/run1/',
              load_path='./logs/layer_inversion/vgg16/l5_dc/run1/ckpt-3000')


spec3 = dict(op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
             op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
             hidden_channels=128, target_shape=[112, 112, 128])

param3 = dict(classifier='vgg16', inv_input_name='conv2_1/relu:0', inv_target_name='pool1:0',
              inv_model_type='deconv_conv',
              inv_model_specs=spec3,
              log_path='./logs/layer_inversion/vgg16/l4_dc/run1/',
              load_path='./logs/layer_inversion/vgg16/l4_dc/run1/ckpt-3000')

spec4 = dict(op1_height=5, op1_width=5, op1_strides=[1, 2, 2, 1],
             op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
             hidden_channels=64)
param4 = dict(classifier='vgg16', inv_input_name='pool1:0', inv_target_name='conv1_2/relu:0',
              inv_model_type='deconv_conv',
              inv_model_specs=spec4,
              log_path='./logs/layer_inversion/vgg16/l3_dc/run1/',
              load_path='./logs/layer_inversion/vgg16/l3_dc/run1/ckpt-3000')

spec5 = dict(op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
             op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
             hidden_channels=64)
param5 = dict(classifier='vgg16', inv_input_name='conv1_2/relu:0', inv_target_name='conv1_1/relu:0',
              inv_model_type='conv_deconv',
              inv_model_specs=spec5,
              log_path='./logs/layer_inversion/vgg16/l2_cd/run1/',
              load_path='./logs/layer_inversion/vgg16/l2_cd/run1/ckpt-3000')

spec6 = dict(op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
             op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
             hidden_channels=64)
param6 = dict(classifier='vgg16', inv_input_name='conv1_1/relu:0', inv_target_name='bgr_normed:0',
              inv_model_type='conv_deconv',
              inv_model_specs=spec6,
              log_path='./logs/layer_inversion/vgg16/l1_cd/run3/',
              load_path='./logs/layer_inversion/vgg16/l1_cd/run3/ckpt-3000')


param1.update(default_params())
param2.update(default_params())
param3.update(default_params())
param4.update(default_params())
param5.update(default_params())
param6.update(default_params())

run_stacked_models([param1, param2, param3, param4, param5, param6])
run_stacked_models([selected_images(param1), selected_images(param2),
                    selected_images(param3), selected_images(param4),
                    selected_images(param5), selected_images(param6)], file_name='stacked_selected')

