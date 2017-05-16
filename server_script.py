from mahendran_vedaldi import invert_layer


params = dict(image_path='./data/selected/images_resized/val13_monkey.bmp', layer_name='conv1/lin:0',
              classifier='alexnet',
              alpha_l2=6, beta_tv=2,
              lambda_l2=2.16e+8, lambda_tv=5e-1,
              sigma=2.7098e+4,
              learning_rate=0.004,
              num_iterations=4000,
              print_freq=50, log_freq=1000, summary_freq=10, lr_lower_freq=500,
              grad_clip=100.0,
              log_path='./logs/mahendran_vedaldi/alexnet/l1/')

invert_layer(params)

params['learning_rate'] = 0.004
params['layer_name'] = 'conv2/relu:0'
params['log_path'] = './logs/mahendran_vedaldi/alexnet/l2/'
invert_layer(params)

params['learning_rate'] = 0.004
params['layer_name'] = 'lrn1:0'
params['log_path'] = './logs/mahendran_vedaldi/alexnet/l3/'
invert_layer(params)

params['learning_rate'] = 0.004
params['layer_name'] = 'pool1:0'
params['log_path'] = './logs/mahendran_vedaldi/alexnet/l4/'
invert_layer(params)

params['learning_rate'] = 0.004
params['layer_name'] = 'conv2/lin:0'
params['log_path'] = './logs/mahendran_vedaldi/alexnet/l5/'
invert_layer(params)

params['learning_rate'] = 0.004
params['layer_name'] = 'conv2/relu:0'
params['log_path'] = './logs/mahendran_vedaldi/alexnet/l6/'
invert_layer(params)

params['lambda_tv'] = 5.0

params['learning_rate'] = 0.004
params['layer_name'] = 'lrn2:0'
params['log_path'] = './logs/mahendran_vedaldi/alexnet/l7/'
invert_layer(params)

params['learning_rate'] = 0.004
params['layer_name'] = 'pool2:0'
params['log_path'] = './logs/mahendran_vedaldi/alexnet/l8/'
invert_layer(params)

params['learning_rate'] = 0.004
params['layer_name'] = 'conv3/lin:0'
params['log_path'] = './logs/mahendran_vedaldi/alexnet/l9/'
invert_layer(params)

params['learning_rate'] = 0.004
params['layer_name'] = 'conv3/relu:0'
params['log_path'] = './logs/mahendran_vedaldi/alexnet/l10/'
invert_layer(params)

params['learning_rate'] = 0.004
params['layer_name'] = 'conv4/lin:0'
params['log_path'] = './logs/mahendran_vedaldi/alexnet/l11/'
invert_layer(params)

params['learning_rate'] = 0.004
params['layer_name'] = 'conv4/relu:0'
params['log_path'] = './logs/mahendran_vedaldi/alexnet/l12/'
invert_layer(params)

params['lambda_tv'] = 50.0

params['learning_rate'] = 0.004
params['layer_name'] = 'conv5/lin:0'
params['log_path'] = './logs/mahendran_vedaldi/alexnet/l13/'
invert_layer(params)

params['learning_rate'] = 0.004
params['layer_name'] = 'conv5/relu:0'
params['log_path'] = './logs/mahendran_vedaldi/alexnet/l14/'
invert_layer(params)

params['learning_rate'] = 0.004
params['layer_name'] = 'pool5:0'
params['log_path'] = './logs/mahendran_vedaldi/alexnet/l15/'
invert_layer(params)

params['learning_rate'] = 0.004
params['layer_name'] = 'fc6/lin:0'
params['log_path'] = './logs/mahendran_vedaldi/alexnet/l16/'
invert_layer(params)

params['learning_rate'] = 0.004
params['layer_name'] = 'fc6/relu:0'
params['log_path'] = './logs/mahendran_vedaldi/alexnet/l17/'
invert_layer(params)

params['learning_rate'] = 0.004
params['layer_name'] = 'fc7/lin:0'
params['log_path'] = './logs/mahendran_vedaldi/alexnet/l18/'
invert_layer(params)

params['learning_rate'] = 0.004
params['layer_name'] = 'fc7/relu:0'
params['log_path'] = './logs/mahendran_vedaldi/alexnet/l19/'
invert_layer(params)

params['learning_rate'] = 0.004
params['layer_name'] = 'fc8/lin:0'
params['log_path'] = './logs/mahendran_vedaldi/alexnet/l20/'
invert_layer(params)


# from net_inversion import NetInversion
# from parameter_utils import default_params
#
#
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
#               log_path='./logs/net_inversion/vgg16/l1-10_10dc/run4/',
#               load_path='./logs/net_inversion/vgg16/l1-10_10dc/run3/ckpt-3000')
# params.update(default_params())
# params['learning_rate'] = 0.00003
#
# print(params)
# ni = NetInversion(params)
# ni.train()

