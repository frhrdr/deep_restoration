from parameter_utils import default_params, selected_images
from deep_restoration.old.layer_inversion import run_stacked_models

specs1 = dict(op1_height=5, op1_width=5, op1_strides=[1, 2, 2, 1],
              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
              hidden_channels=64)
params1 = dict(classifier='vgg16', inv_input_name='pool1:0', inv_target_name='conv1_2/relu:0',
               inv_model_type='deconv_conv',
               inv_model_specs=specs1,
               log_path='./logs/layer_inversion/vgg16/l3_dc/run1/',
               load_path='./logs/layer_inversion/vgg16/l3_dc/run1/ckpt-3000')

specs2 = dict(op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
              hidden_channels=64)
params2 = dict(classifier='vgg16', inv_input_name='conv1_2/relu:0', inv_target_name='conv1_1/relu:0',
               inv_model_type='conv_deconv',
               inv_model_specs=specs2,
               log_path='./logs/layer_inversion/vgg16/l2_cd/run1/',
               load_path='./logs/layer_inversion/vgg16/l2_cd/run1/ckpt-3000')

specs3 = dict(op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
              hidden_channels=64)
params3 = dict(classifier='vgg16', inv_input_name='conv1_1/relu:0', inv_target_name='bgr_normed:0',
               inv_model_type='conv_deconv',
               inv_model_specs=specs3,
               log_path='./logs/layer_inversion/vgg16/l1_cd/run3/',
               load_path='./logs/layer_inversion/vgg16/l1_cd/run3/ckpt-3000')

params1.update(default_params())
params2.update(default_params())
params3.update(default_params())

run_stacked_models([params1, params2, params3])
run_stacked_models([selected_images(params1), selected_images(params2), selected_images(params3)], file_name='stacked_selected')
