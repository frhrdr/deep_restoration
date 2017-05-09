from net_inversion import NetInversion, run_stacked_models
# from layer_inversion import LayerInversion, run_stacked_models
from parameter_utils import default_params, selected_images

spec1 = dict(inv_model_type='deconv_conv',
             op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
             op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
             hidden_channels=64, target_shape=[224, 224, 3],
             inv_input_name='conv1_1/relu:0', inv_target_name='bgr_normed:0',
             rec_name='pool2_rec', add_loss=True)
#
# spec2 = dict(inv_model_type='deconv_conv',
#              op1_height=5, op1_width=5, op1_strides=[1, 4, 4, 1],
#              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
#              hidden_channels=256, target_shape=[56, 56, 128],
#              inv_input_name='module_0/pool2_rec:0', inv_target_name='bgr_normed:0',
#              rec_name='reconstruction', add_loss=True)


params = dict(classifier='vgg16',
              inv_model_type='deconv_conv',
              inv_model_specs=[spec1],
              log_path='./logs/net_inversion/vgg16/test/',
              load_path='./logs/layer_inversion/vgg16/l1_dc/run1/ckpt-3000')

params.update(default_params())
params['batch_size'] = 2

print(params)
ni = NetInversion(params)
ni.train()

# spec1 = dict(op1_height=5, op1_width=5, op1_strides=[1, 2, 2, 1],
#              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
#              hidden_channels=128, target_shape=[112, 112, 128])
#
# param1 = dict(classifier='vgg16', inv_input_name='pool2:0', inv_target_name='conv2_2/relu:0',
#               inv_model_type='deconv_conv',
#               inv_model_specs=spec1,
#               log_path='./logs/layer_inversion/vgg16/l6_dc/run1/',
#               load_path='./logs/layer_inversion/vgg16/l6_dc/run1/ckpt-3000')
#
# param1.update(default_params())


print('l123' in '_ksefl123_00')