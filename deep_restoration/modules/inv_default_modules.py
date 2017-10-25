from modules.inv_modules import DeconvConvModule


def alexnet_inv():
    inv = dict()

    inv['DC9'] = DeconvConvModule(inv_input_name='conv5/lin:0', inv_target_name='conv4/lin:0',
                                  hidden_channels=384, rec_name='c4l_rec',
                                  op1_hw=[8, 8], op1_strides=[1, 1, 1, 1], op2_hw=[8, 8], op2_strides=[1, 1, 1, 1],
                                  name='DC7', subdir='solotrain', trainable=True)

    inv['DC8'] = DeconvConvModule(inv_input_name='conv4/lin:0', inv_target_name='conv3/lin:0',
                                  hidden_channels=384, rec_name='c3l_rec',
                                  op1_hw=[8, 8], op1_strides=[1, 1, 1, 1], op2_hw=[8, 8], op2_strides=[1, 1, 1, 1],
                                  name='DC7', subdir='solotrain', trainable=True)

    inv['DC7'] = DeconvConvModule(inv_input_name='conv3/lin:0', inv_target_name='pool2:0',
                                  hidden_channels=384, rec_name='pool2_rec',
                                  op1_hw=[8, 8], op1_strides=[1, 1, 1, 1], op2_hw=[8, 8], op2_strides=[1, 1, 1, 1],
                                  name='DC7', subdir='solotrain', trainable=True)

    inv['DC6'] = DeconvConvModule(inv_input_name='pool2:0', inv_target_name='lrn2:0',
                                  hidden_channels=256, rec_name='lrn2_rec',
                                  op1_hw=[3, 3], op1_strides=[1, 2, 2, 1], op2_hw=[8, 8], op2_strides=[1, 1, 1, 1],
                                  op1_pad='VALID', op2_pad='SAME',
                                  name='DC6', subdir='solotrain', trainable=True)

    inv['DC5'] = DeconvConvModule(inv_input_name='lrn2:0', inv_target_name='conv2/lin:0',
                                  hidden_channels=256, rec_name='c2l_rec',
                                  op1_hw=[8, 8], op1_strides=[1, 1, 1, 1], op2_hw=[8, 8], op2_strides=[1, 1, 1, 1],
                                  name='DC5', subdir='solotrain', trainable=True)

    inv['DC4'] = DeconvConvModule(inv_input_name='conv2/lin:0', inv_target_name='pool1:0',
                                  hidden_channels=256, rec_name='pool1_rec',
                                  op1_hw=[8, 8], op1_strides=[1, 1, 1, 1], op2_hw=[8, 8], op2_strides=[1, 1, 1, 1],
                                  name='DC4', subdir='solotrain', trainable=True)

    inv['DC3'] = DeconvConvModule(inv_input_name='pool1:0', inv_target_name='lrn1:0',
                                  hidden_channels=96, rec_name='lrn1_rec',
                                  op1_hw=[3, 3], op1_strides=[1, 2, 2, 1], op2_hw=[8, 8], op2_strides=[1, 1, 1, 1],
                                  op1_pad='VALID', op2_pad='SAME',
                                  name='DC3', subdir='solotrain', trainable=True)

    inv['DC2'] = DeconvConvModule(inv_input_name='lrn1:0', inv_target_name='conv1/lin:0',
                                  hidden_channels=96, rec_name='c1l_rec',
                                  op1_hw=[8, 8], op1_strides=[1, 1, 1, 1], op2_hw=[8, 8], op2_strides=[1, 1, 1, 1],
                                  name='DC2', subdir='solotrain', trainable=True)

    inv['DC1'] = DeconvConvModule(inv_input_name='conv1/lin:0', inv_target_name='rgb_scaled:0',
                                  hidden_channels=96, rec_name='rgb_rec',
                                  op1_hw=[11, 11], op1_strides=[1, 4, 4, 1], op2_hw=[11, 11], op2_strides=[1, 1, 1, 1],
                                  op1_pad='VALID', name='DC1', subdir='solotrain', trainable=True)

    return inv


def vgg16_inv():
    inv = dict()

    spec1 = dict(op1_height=5, op1_width=5, op1_strides=[1, 2, 2, 1],
                 op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
                 hidden_channels=256, target_shape=[28, 28, 256])

    param1 = dict(classifier='vgg16', inv_input_name='pool3:0', inv_target_name='conv3_3/relu:0',
                  inv_model_type='deconv_conv',
                  inv_model_specs=spec1,
                  log_path='./logs/layer_inversion/vgg16/l10_dc/1e-13/',
                  load_path='./logs/layer_inversion/vgg16/l10_dc/1e-13/ckpt-5000')

    spec2 = dict(op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
                 op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
                 hidden_channels=256, target_shape=[56, 56, 256])

    param2 = dict(classifier='vgg16', inv_input_name='conv3_3/relu:0', inv_target_name='conv3_2/relu:0',
                  inv_model_type='deconv_conv',
                  inv_model_specs=spec2,
                  log_path='./logs/layer_inversion/vgg16/l9_dc/1e-13/',
                  load_path='./logs/layer_inversion/vgg16/l9_dc/1e-13/ckpt-5000')

    spec3 = dict(op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
                 op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
                 hidden_channels=256, target_shape=[56, 56, 256])

    param3 = dict(classifier='vgg16', inv_input_name='conv3_2/relu:0', inv_target_name='conv3_1/relu:0',
                  inv_model_type='deconv_conv',
                  inv_model_specs=spec3,
                  log_path='./logs/layer_inversion/vgg16/l8_dc/1e-13/',
                  load_path='./logs/layer_inversion/vgg16/l8_dc/1e-13/ckpt-5000')

    spec4 = dict(op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
                 op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
                 hidden_channels=256, target_shape=[56, 56, 128])

    param4 = dict(classifier='vgg16', inv_input_name='conv3_1/relu:0', inv_target_name='pool2:0',
                  inv_model_type='deconv_conv',
                  inv_model_specs=spec4,
                  log_path='./logs/layer_inversion/vgg16/l7_dc/1e-13/',
                  load_path='./logs/layer_inversion/vgg16/l7_dc/1e-13/ckpt-5000')

    spec5 = dict(op1_height=5, op1_width=5, op1_strides=[1, 2, 2, 1],
                 op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
                 hidden_channels=128, target_shape=[112, 112, 128])

    param5 = dict(classifier='vgg16', inv_input_name='pool2:0', inv_target_name='conv2_2/relu:0',
                  inv_model_type='deconv_conv',
                  inv_model_specs=spec5,
                  log_path='./logs/layer_inversion/vgg16/l6_dc/1e-13/',
                  load_path='./logs/layer_inversion/vgg16/l6_dc/1e-13/ckpt-3000')

    spec6 = dict(op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
                 op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
                 hidden_channels=128, target_shape=[112, 112, 128])

    param6 = dict(classifier='vgg16', inv_input_name='conv2_2/relu:0', inv_target_name='conv2_1/relu:0',
                  inv_model_type='deconv_conv',
                  inv_model_specs=spec6,
                  log_path='./logs/layer_inversion/vgg16/l5_dc/1e-13/',
                  load_path='./logs/layer_inversion/vgg16/l5_dc/1e-13/ckpt-3000')

    spec7 = dict(op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
                 op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
                 hidden_channels=128, target_shape=[112, 112, 128])

    param7 = dict(classifier='vgg16', inv_input_name='conv2_1/relu:0', inv_target_name='pool1:0',
                  inv_model_type='deconv_conv',
                  inv_model_specs=spec7,
                  log_path='./logs/layer_inversion/vgg16/l4_dc/1e-13/',
                  load_path='./logs/layer_inversion/vgg16/l4_dc/1e-13/ckpt-3000')

    spec8 = dict(op1_height=5, op1_width=5, op1_strides=[1, 2, 2, 1],
                 op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
                 hidden_channels=64)
    param8 = dict(classifier='vgg16', inv_input_name='pool1:0', inv_target_name='conv1_2/relu:0',
                  inv_model_type='deconv_conv',
                  inv_model_specs=spec8,
                  log_path='./logs/layer_inversion/vgg16/l3_dc/1e-13/',
                  load_path='./logs/layer_inversion/vgg16/l3_dc/1e-13/ckpt-3000')

    spec9 = dict(op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
                 op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
                 hidden_channels=64)
    param9 = dict(classifier='vgg16', inv_input_name='conv1_2/relu:0', inv_target_name='conv1_1/relu:0',
                  inv_model_type='conv_deconv',
                  inv_model_specs=spec9,
                  log_path='./logs/layer_inversion/vgg16/l2_cd/1e-13/',
                  load_path='./logs/layer_inversion/vgg16/l2_cd/1e-13/ckpt-3000')

    spec10 = dict(op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
                  op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
                  hidden_channels=64)
    param10 = dict(classifier='vgg16', inv_input_name='conv1_1/relu:0', inv_target_name='bgr_normed:0',
                   inv_model_type='conv_deconv',
                   inv_model_specs=spec10,
                   log_path='./logs/layer_inversion/vgg16/l1_cd/run3/',
                   load_path='./logs/layer_inversion/vgg16/l1_cd/run3/ckpt-3000')

    inv['DC1'] = DeconvConvModule(inv_input_name='conv1_1/lin:0', inv_target_name='rgb_scaled:0',
                                  hidden_channels=66, rec_name='rgb_rec',
                                  op1_hw=[5, 5], op1_strides=[1, 1, 1, 1], op2_hw=[5, 5], op2_strides=[1, 1, 1, 1],
                                  name='DC1', subdir='solotrain', trainable=True)
