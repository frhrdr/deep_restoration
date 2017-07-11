from parameter_utils import default_params, selected_images
from deep_restoration.net_inversion import run_stacked_models

spec1 = dict(inv_model_type='deconv_conv',
             op1_height=3, op1_width=3, op1_strides=[1, 2, 2, 1], op1_pad='VALID',
             op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1], op2_pad='SAME',
             hidden_channels=96, target_shape=[55, 55, 96],
             inv_input_name='pool1:0', inv_target_name='conv1/relu:0',
             rec_name='reconstruction', add_loss=True)

param1 = dict(classifier='alexnet',
              inv_model_type='deconv_conv',
              inv_model_specs=[spec1],
              log_path='./logs/net_inversion/alexnet/l2_dc/run1/',
              load_path='')

spec2 = dict(inv_model_type='deconv_conv',
             op1_height=11, op1_width=11, op1_strides=[1, 4, 4, 1], op1_pad='SAME',
             op2_height=11, op2_width=11, op2_strides=[1, 1, 1, 1], op2_pad='SAME',
             hidden_channels=96, target_shape=[224, 224, 3],
             inv_input_name='conv1/relu:0', inv_target_name='bgr_normed:0',
             rec_name='reconstruction', add_loss=True)

param2 = dict(classifier='alexnet',
              inv_model_type='deconv_conv',
              inv_model_specs=[spec2],
              log_path='./logs/net_inversion/alexnet/l1_dc/run1/',
              ckpt_num=500,
              load_path='')


param1.update(default_params())
param2.update(default_params())

run_stacked_models([param1, param2])
run_stacked_models([selected_images(param1), selected_images(param2)], file_name='stacked_selected')
