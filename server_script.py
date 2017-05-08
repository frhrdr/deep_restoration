from layer_inversion import LayerInversion
from parameter_utils import default_params

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


specs = dict(op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
             op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
             hidden_channels=256, target_shape=[56, 56, 128])

params = dict(classifier='vgg16', inv_input_name='conv3_1/relu:0', inv_target_name='pool2:0',
              inv_model_type='deconv_conv',
              inv_model_specs=specs,
              log_path='./logs/layer_inversion/vgg16/l7_dc/run1/',
              load_path='')
params.update(default_params())
print(params)
li = LayerInversion(params)
li.train()


specs = dict(op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
             op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
             hidden_channels=256, target_shape=[56, 56, 256])

params = dict(classifier='vgg16', inv_input_name='conv3_2/relu:0', inv_target_name='conv3_1/relu:0',
              inv_model_type='deconv_conv',
              inv_model_specs=specs,
              log_path='./logs/layer_inversion/vgg16/l8_dc/run1/',
              load_path='')
params.update(default_params())
print(params)
li = LayerInversion(params)
li.train()


specs = dict(op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
             op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
             hidden_channels=256, target_shape=[56, 56, 256])

params = dict(classifier='vgg16', inv_input_name='conv3_3/relu:0', inv_target_name='conv3_2/relu:0',
              inv_model_type='deconv_conv',
              inv_model_specs=specs,
              log_path='./logs/layer_inversion/vgg16/l9_dc/run1/',
              load_path='')
params.update(default_params())
print(params)
li = LayerInversion(params)
li.train()


specs = dict(op1_height=5, op1_width=5, op1_strides=[1, 2, 2, 1],
             op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
             hidden_channels=256, target_shape=[28, 28, 256])

params = dict(classifier='vgg16', inv_input_name='pool3:0', inv_target_name='conv3_3/relu:0',
              inv_model_type='deconv_conv',
              inv_model_specs=specs,
              log_path='./logs/layer_inversion/vgg16/l10_dc/run1/',
              load_path='')
params.update(default_params())
print(params)
li = LayerInversion(params)
li.train()