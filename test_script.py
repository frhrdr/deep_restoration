from net_inversion import NetInversion, run_stacked_models
from parameter_utils import default_params, selected_images


spec1 = dict(inv_model_type='deconv_conv',
             op1_height=3, op1_width=3, op1_strides=[1, 2, 2, 1], op1_pad='VALID',
             op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1], op2_pad='SAME',
             hidden_channels=96, target_shape=[55, 55, 96],
             inv_input_name='pool1:0', inv_target_name='conv1/relu:0',
             rec_name='conv1_rec', add_loss=True)

spec2 = dict(inv_model_type='deconv_conv',
             op1_height=11, op1_width=11, op1_strides=[1, 4, 4, 1], op1_pad='SAME',
             op2_height=11, op2_width=11, op2_strides=[1, 1, 1, 1], op2_pad='SAME',
             hidden_channels=96, target_shape=[224, 224, 96],
             inv_input_name='module_1/conv1_rec:0', inv_target_name='bgr_normed:0',
             rec_name='reconstruction', add_loss=True)

params = dict(classifier='alexnet',
              inv_model_specs=[spec1, spec2],
              log_path='./logs/net_inversion/alexnet/l12_2dc/run3/',
              load_path='')
params.update(default_params())
print(params)
ni = NetInversion(params)
ni.visualize()

ni = NetInversion(selected_images(params))
ni.visualize(file_name='selected_diff')