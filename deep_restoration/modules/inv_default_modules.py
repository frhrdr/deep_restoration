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
    pass