# from layer_inversion import LayerInversion
from net_inversion import NetInversion
from parameter_utils import default_params


spec1 = dict(inv_model_type='deconv_conv',
             op1_height=11, op1_width=11, op1_strides=[1, 4, 4, 1],
             op2_height=11, op2_width=11, op2_strides=[1, 1, 1, 1],
             hidden_channels=96, target_shape=[224, 224, 3],
             inv_input_name='conv1/relu:0', inv_target_name='bgr_normed:0',
             rec_name='reconstruction', add_loss=True)

params = dict(classifier='alexnet',
              inv_model_type='deconv_conv',
              inv_model_specs=[spec1],
              log_path='./logs/net_inversion/alexnet/l1_dc/run1/',
              load_path='./logs/layer_inversion/alexnet/l1_dc/run2/ckpt-3000')

params.update(default_params())
params['num_iterations'] = 500
print(params)

ni = NetInversion(params)
ni.train()










# specs = dict(op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
#              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
#              hidden_channels=128, target_shape=[112, 112, 128])
#
# params = dict(classifier='vgg16', inv_input_name='conv2_1/relu:0', inv_target_name='pool1:0',
#               inv_model_type='deconv_conv',
#               inv_model_specs=specs,
#               log_path='./logs/layer_inversion/vgg16/l4_dc/run1/',
#               load_path='')
# params.update(default_params())
# print(params)
# li = LayerInversion(params)
# li.train()
#
#
# specs = dict(op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
#              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
#              hidden_channels=128, target_shape=[112, 112, 128])
#
# params = dict(classifier='vgg16', inv_input_name='conv2_2/relu:0', inv_target_name='conv2_1/relu:0',
#               inv_model_type='deconv_conv',
#               inv_model_specs=specs,
#               log_path='./logs/layer_inversion/vgg16/l5_dc/run1/',
#               load_path='')
# params.update(default_params())
# print(params)
# li = LayerInversion(params)
# li.train()
#
#
# specs = dict(op1_height=5, op1_width=5, op1_strides=[1, 2, 2, 1],
#              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
#              hidden_channels=128, target_shape=[112, 112, 128])
#
# params = dict(classifier='vgg16', inv_input_name='pool2:0', inv_target_name='conv2_2/relu:0',
#               inv_model_type='deconv_conv',
#               inv_model_specs=specs,
#               log_path='./logs/layer_inversion/vgg16/l6_dc/run1/',
#               load_path='')
# params.update(default_params())
# print(params)
# li = LayerInversion(params)
# li.train()
#
#
# specs = dict(op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
#              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
#              hidden_channels=256, target_shape=[56, 56, 128])
#
# params = dict(classifier='vgg16', inv_input_name='conv3_1/relu:0', inv_target_name='pool2:0',
#               inv_model_type='deconv_conv',
#               inv_model_specs=specs,
#               log_path='./logs/layer_inversion/vgg16/l7_dc/run1/',
#               load_path='')
# params.update(default_params())
# params['num_iterations'] = 5000
# print(params)
# li = LayerInversion(params)
# li.train()
#
#
# specs = dict(op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
#              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
#              hidden_channels=256, target_shape=[56, 56, 256])
#
# params = dict(classifier='vgg16', inv_input_name='conv3_2/relu:0', inv_target_name='conv3_1/relu:0',
#               inv_model_type='deconv_conv',
#               inv_model_specs=specs,
#               log_path='./logs/layer_inversion/vgg16/l8_dc/run1/',
#               load_path='')
# params.update(default_params())
# params['num_iterations'] = 5000
# print(params)
# li = LayerInversion(params)
# li.train()
#
#
# specs = dict(op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
#              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
#              hidden_channels=256, target_shape=[56, 56, 256])
#
# params = dict(classifier='vgg16', inv_input_name='conv3_3/relu:0', inv_target_name='conv3_2/relu:0',
#               inv_model_type='deconv_conv',
#               inv_model_specs=specs,
#               log_path='./logs/layer_inversion/vgg16/l9_dc/run1/',
#               load_path='')
# params.update(default_params())
# params['num_iterations'] = 5000
# print(params)
# li = LayerInversion(params)
# li.train()
#
#
# specs = dict(op1_height=5, op1_width=5, op1_strides=[1, 2, 2, 1],
#              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
#              hidden_channels=256, target_shape=[28, 28, 256])
#
# params = dict(classifier='vgg16', inv_input_name='pool3:0', inv_target_name='conv3_3/relu:0',
#               inv_model_type='deconv_conv',
#               inv_model_specs=specs,
#               log_path='./logs/layer_inversion/vgg16/l10_dc/run1/',
#               load_path='')
# params.update(default_params())
# params['num_iterations'] = 5000
# print(params)
# li = LayerInversion(params)
# li.train()