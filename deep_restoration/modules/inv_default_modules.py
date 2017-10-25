from modules.inv_modules import DeconvConvModule


def alexnet_inv(subdir='solotrain', alt_input=None, alt_load_subdir=None):
    inv = dict()

    inv['DC9'] = DeconvConvModule(classifier='alexnet', inv_input_name='conv5/lin:0', inv_target_name='conv4/lin:0',
                                  hidden_channels=384, rec_name='c4l_rec',
                                  op1_hw=[8, 8], op1_strides=[1, 1, 1, 1], op2_hw=[8, 8], op2_strides=[1, 1, 1, 1], 
                                  input_from_rec=alt_input,
                                  name='DC7', subdir=subdir, trainable=True, alt_load_subdir=alt_load_subdir)

    inv['DC8'] = DeconvConvModule(classifier='alexnet', inv_input_name='conv4/lin:0', inv_target_name='conv3/lin:0',
                                  hidden_channels=384, rec_name='c3l_rec',
                                  op1_hw=[8, 8], op1_strides=[1, 1, 1, 1], op2_hw=[8, 8], op2_strides=[1, 1, 1, 1], 
                                  input_from_rec=alt_input,
                                  name='DC7', subdir=subdir, trainable=True, alt_load_subdir=alt_load_subdir)

    inv['DC7'] = DeconvConvModule(classifier='alexnet', inv_input_name='conv3/lin:0', inv_target_name='pool2:0',
                                  hidden_channels=384, rec_name='pool2_rec',
                                  op1_hw=[8, 8], op1_strides=[1, 1, 1, 1], op2_hw=[8, 8], op2_strides=[1, 1, 1, 1], 
                                  input_from_rec=alt_input,
                                  name='DC7', subdir=subdir, trainable=True, alt_load_subdir=alt_load_subdir)

    inv['DC6'] = DeconvConvModule(classifier='alexnet', inv_input_name='pool2:0', inv_target_name='lrn2:0',
                                  hidden_channels=256, rec_name='lrn2_rec',
                                  op1_hw=[3, 3], op1_strides=[1, 2, 2, 1], op2_hw=[8, 8], op2_strides=[1, 1, 1, 1], 
                                  input_from_rec=alt_input,
                                  op1_pad='VALID', op2_pad='SAME',
                                  name='DC6', subdir=subdir, trainable=True, alt_load_subdir=alt_load_subdir)

    inv['DC5'] = DeconvConvModule(classifier='alexnet', inv_input_name='lrn2:0', inv_target_name='conv2/lin:0',
                                  hidden_channels=256, rec_name='c2l_rec',
                                  op1_hw=[8, 8], op1_strides=[1, 1, 1, 1], op2_hw=[8, 8], op2_strides=[1, 1, 1, 1], 
                                  input_from_rec=alt_input,
                                  name='DC5', subdir=subdir, trainable=True, alt_load_subdir=alt_load_subdir)

    inv['DC4'] = DeconvConvModule(classifier='alexnet', inv_input_name='conv2/lin:0', inv_target_name='pool1:0',
                                  hidden_channels=256, rec_name='pool1_rec',
                                  op1_hw=[8, 8], op1_strides=[1, 1, 1, 1], op2_hw=[8, 8], op2_strides=[1, 1, 1, 1], 
                                  input_from_rec=alt_input,
                                  name='DC4', subdir=subdir, trainable=True, alt_load_subdir=alt_load_subdir)

    inv['DC3'] = DeconvConvModule(classifier='alexnet', inv_input_name='pool1:0', inv_target_name='lrn1:0',
                                  hidden_channels=96, rec_name='lrn1_rec',
                                  op1_hw=[3, 3], op1_strides=[1, 2, 2, 1], op2_hw=[8, 8], op2_strides=[1, 1, 1, 1], 
                                  input_from_rec=alt_input,
                                  op1_pad='VALID', op2_pad='SAME',
                                  name='DC3', subdir=subdir, trainable=True, alt_load_subdir=alt_load_subdir)

    inv['DC2'] = DeconvConvModule(classifier='alexnet', inv_input_name='lrn1:0', inv_target_name='conv1/lin:0',
                                  hidden_channels=96, rec_name='c1l_rec',
                                  op1_hw=[8, 8], op1_strides=[1, 1, 1, 1], op2_hw=[8, 8], op2_strides=[1, 1, 1, 1], 
                                  input_from_rec=alt_input,
                                  name='DC2', subdir=subdir, trainable=True, alt_load_subdir=alt_load_subdir)

    inv['DC1'] = DeconvConvModule(classifier='alexnet', inv_input_name='conv1/lin:0', inv_target_name='rgb_scaled:0',
                                  hidden_channels=96, rec_name='rgb_rec',
                                  op1_hw=[11, 11], op1_strides=[1, 4, 4, 1], op2_hw=[11, 11], op2_strides=[1, 1, 1, 1], 
                                  input_from_rec=alt_input,
                                  op1_pad='VALID', name='DC1', subdir=subdir, trainable=True,
                                  alt_load_subdir=alt_load_subdir)

    return inv


def vgg16_inv(subdir='solotrain', alt_input=None, alt_load_subdir=None):
    inv = dict()
    
    inv['DC10'] = DeconvConvModule(classifier='vgg16', inv_input_name='pool3:0', inv_target_name='conv3_3/lin:0',
                                   hidden_channels=256, rec_name='c33l_rec',
                                   op1_hw=[5, 5], op1_strides=[1, 2, 2, 1], op2_hw=[5, 5], op2_strides=[1, 1, 1, 1], 
                                   input_from_rec=alt_input,
                                   name='DC10', subdir=subdir, trainable=True, alt_load_subdir=alt_load_subdir)

    inv['DC9'] = DeconvConvModule(classifier='vgg16', inv_input_name='conv3_3/lin:0', inv_target_name='conv3_2/lin:0',
                                  hidden_channels=256, rec_name='c32l_rec',
                                  op1_hw=[5, 5], op1_strides=[1, 1, 1, 1], op2_hw=[5, 5], op2_strides=[1, 1, 1, 1], 
                                  input_from_rec=alt_input,
                                  name='DC9', subdir=subdir, trainable=True, alt_load_subdir=alt_load_subdir)

    inv['DC8'] = DeconvConvModule(classifier='vgg16', inv_input_name='conv3_2/lin:0', inv_target_name='conv3_1/lin:0',
                                  hidden_channels=256, rec_name='c31l_rec',
                                  op1_hw=[5, 5], op1_strides=[1, 1, 1, 1], op2_hw=[5, 5], op2_strides=[1, 1, 1, 1], 
                                  input_from_rec=alt_input,
                                  name='DC8', subdir=subdir, trainable=True, alt_load_subdir=alt_load_subdir)

    inv['DC7'] = DeconvConvModule(classifier='vgg16', inv_input_name='conv3_1/lin:0', inv_target_name='pool2:0',
                                  hidden_channels=256, rec_name='pool2_rec',
                                  op1_hw=[5, 5], op1_strides=[1, 1, 1, 1], op2_hw=[5, 5], op2_strides=[1, 1, 1, 1], 
                                  input_from_rec=alt_input,
                                  name='DC7', subdir=subdir, trainable=True, alt_load_subdir=alt_load_subdir)
    
    inv['DC6'] = DeconvConvModule(classifier='vgg16', inv_input_name='pool2:0', inv_target_name='conv2_2/lin:0',
                                  hidden_channels=128, rec_name='c22l_rec',
                                  op1_hw=[5, 5], op1_strides=[1, 2, 2, 1], op2_hw=[5, 5], op2_strides=[1, 1, 1, 1], 
                                  input_from_rec=alt_input,
                                  name='DC6', subdir=subdir, trainable=True, alt_load_subdir=alt_load_subdir)

    inv['DC5'] = DeconvConvModule(classifier='vgg16', inv_input_name='conv2_2/lin:0', inv_target_name='conv2_1/lin:0',
                                  hidden_channels=128, rec_name='c21l_rec',
                                  op1_hw=[5, 5], op1_strides=[1, 1, 1, 1], op2_hw=[5, 5], op2_strides=[1, 1, 1, 1], 
                                  input_from_rec=alt_input,
                                  name='DC5', subdir=subdir, trainable=True, alt_load_subdir=alt_load_subdir)

    inv['DC4'] = DeconvConvModule(classifier='vgg16', inv_input_name='conv2_1/lin:0', inv_target_name='pool1:0',
                                  hidden_channels=128, rec_name='pool1_rec',
                                  op1_hw=[5, 5], op1_strides=[1, 1, 1, 1], op2_hw=[5, 5], op2_strides=[1, 1, 1, 1], 
                                  input_from_rec=alt_input,
                                  name='DC4', subdir=subdir, trainable=True, alt_load_subdir=alt_load_subdir)

    inv['DC3'] = DeconvConvModule(classifier='vgg16', inv_input_name='pool1:0', inv_target_name='conv1_2/lin:0',
                                  hidden_channels=64, rec_name='c12l_rec',
                                  op1_hw=[5, 5], op1_strides=[1, 2, 2, 1], op2_hw=[5, 5], op2_strides=[1, 1, 1, 1], 
                                  input_from_rec=alt_input,
                                  name='DC3', subdir=subdir, trainable=True, alt_load_subdir=alt_load_subdir)

    inv['DC2'] = DeconvConvModule(classifier='vgg16', inv_input_name='conv1_2/lin:0', inv_target_name='conv1_1/lin:0',
                                  hidden_channels=64, rec_name='c11l_rec',
                                  op1_hw=[5, 5], op1_strides=[1, 1, 1, 1], op2_hw=[5, 5], op2_strides=[1, 1, 1, 1], 
                                  input_from_rec=alt_input,
                                  name='DC2', subdir=subdir, trainable=True, alt_load_subdir=alt_load_subdir)

    inv['DC1'] = DeconvConvModule(classifier='vgg16', inv_input_name='conv1_1/lin:0', inv_target_name='rgb_scaled:0',
                                  hidden_channels=64, rec_name='rgb_rec',
                                  op1_hw=[5, 5], op1_strides=[1, 1, 1, 1], op2_hw=[5, 5], op2_strides=[1, 1, 1, 1], 
                                  input_from_rec=alt_input,
                                  name='DC1', subdir=subdir, trainable=True, alt_load_subdir=alt_load_subdir)

    return inv
