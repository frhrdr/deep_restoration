from net_inversion import NetInversion
from parameter_utils import default_params


spec1 = dict(inv_model_type='deconv_conv',
             op1_height=3, op1_width=3, op1_strides=[1, 2, 2, 1], op1_pad='VALID',
             op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1], op2_pad='SAME',
             hidden_channels=96, target_shape=[55, 55, 96],
             inv_input_name='pool1:0', inv_target_name='conv1/relu:0',
             rec_name='reconstruction', add_loss=True)

params = dict(classifier='alexnet',
              inv_model_type='deconv_conv',
              inv_model_specs=[spec1],
              log_path='./logs/net_inversion/alexnet/l2_dc/run1/',
              load_path='')

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
