from modules.inv_modules import DeconvConvModule


def get_stacked_module(classifier, start_layer, rec_layer,
                       alt_load_subdir='solotrain', subdir_name=None, trainable=True):
    assert start_layer > rec_layer
    subdir_name = subdir_name or '{}_stack_{}_to_{}'.format(classifier, start_layer, rec_layer)
    alt_input = None
    module_list = []
    for module_id in range(start_layer, rec_layer - 1, -1):
        dc_module = default_deconv_conv_module(classifier=classifier, module_id=module_id,
                                               subdir=subdir_name, alt_input=alt_input,
                                               alt_load_subdir=alt_load_subdir, trainable=trainable)
        alt_input = dc_module.rec_name + ':0'
        module_list.append(dc_module)
    return module_list


def default_deconv_conv_module(classifier, module_id, subdir='solotrain', 
                               alt_input=None, alt_load_subdir=None, trainable=True):
    assert classifier in ('alexnet', 'vgg16')
    if isinstance(module_id, int):
        module_id = 'DC{}'.format(module_id)
    if classifier == 'alexnet':
        module_specs = alexnet_specs(subdir=subdir, alt_input=alt_input, 
                                     alt_load_subdir=alt_load_subdir, trainable=trainable)[module_id]
    else:
        module_specs = vgg16_specs(subdir=subdir, alt_input=alt_input, 
                                   alt_load_subdir=alt_load_subdir, trainable=trainable)[module_id]
    return DeconvConvModule(**module_specs)


def alexnet_specs(subdir='solotrain', alt_input=None, alt_load_subdir=None, trainable=True):
    inv = dict()

    inv['DC9'] = {'classifier': 'alexnet', 'inv_input_name': 'conv5/lin:0', 'inv_target_name': 'conv4/lin:0',
                  'hidden_channels': 384, 'rec_name': 'c4l_rec',
                  'op1_hw': [8, 8], 'op1_strides': [1, 1, 1, 1], 'op2_hw': [8, 8], 'op2_strides': [1, 1, 1, 1],
                  'input_from_rec': alt_input,
                  'name': 'DC9', 'subdir': subdir, 'trainable': trainable, 'alt_load_subdir': alt_load_subdir}

    inv['DC8'] = {'classifier': 'alexnet', 'inv_input_name': 'conv4/lin:0', 'inv_target_name': 'conv3/lin:0',
                  'hidden_channels': 384, 'rec_name': 'c3l_rec',
                  'op1_hw': [8, 8], 'op1_strides': [1, 1, 1, 1], 'op2_hw': [8, 8], 'op2_strides': [1, 1, 1, 1],
                  'input_from_rec': alt_input,
                  'name': 'DC8', 'subdir': subdir, 'trainable': trainable, 'alt_load_subdir': alt_load_subdir}

    inv['DC7'] = {'classifier': 'alexnet', 'inv_input_name': 'conv3/lin:0', 'inv_target_name': 'pool2:0',
                  'hidden_channels': 384, 'rec_name': 'pool2_rec',
                  'op1_hw': [8, 8], 'op1_strides': [1, 1, 1, 1], 'op2_hw': [8, 8], 'op2_strides': [1, 1, 1, 1],
                  'input_from_rec': alt_input,
                  'name': 'DC7', 'subdir': subdir, 'trainable': trainable, 'alt_load_subdir': alt_load_subdir}

    inv['DC6'] = {'classifier': 'alexnet', 'inv_input_name': 'pool2:0', 'inv_target_name': 'lrn2:0',
                  'hidden_channels': 256, 'rec_name': 'lrn2_rec',
                  'op1_hw': [3, 3], 'op1_strides': [1, 2, 2, 1], 'op2_hw': [8, 8], 'op2_strides': [1, 1, 1, 1],
                  'input_from_rec': alt_input,
                  'op1_pad': 'VALID', 'op2_pad': 'SAME',
                  'name': 'DC6', 'subdir': subdir, 'trainable': trainable, 'alt_load_subdir': alt_load_subdir}

    inv['DC5'] = {'classifier': 'alexnet', 'inv_input_name': 'lrn2:0', 'inv_target_name': 'conv2/lin:0',
                  'hidden_channels': 256, 'rec_name': 'c2l_rec',
                  'op1_hw': [8, 8], 'op1_strides': [1, 1, 1, 1], 'op2_hw': [8, 8], 'op2_strides': [1, 1, 1, 1],
                  'input_from_rec': alt_input,
                  'name': 'DC5', 'subdir': subdir, 'trainable': trainable, 'alt_load_subdir': alt_load_subdir}

    inv['DC4'] = {'classifier': 'alexnet', 'inv_input_name': 'conv2/lin:0', 'inv_target_name': 'pool1:0',
                  'hidden_channels': 256, 'rec_name': 'pool1_rec',
                  'op1_hw': [8, 8], 'op1_strides': [1, 1, 1, 1], 'op2_hw': [8, 8], 'op2_strides': [1, 1, 1, 1],
                  'input_from_rec': alt_input,
                  'name': 'DC4', 'subdir': subdir, 'trainable': trainable, 'alt_load_subdir': alt_load_subdir}

    inv['DC3'] = {'classifier': 'alexnet', 'inv_input_name': 'pool1:0', 'inv_target_name': 'lrn1:0',
                  'hidden_channels': 96, 'rec_name': 'lrn1_rec',
                  'op1_hw': [3, 3], 'op1_strides': [1, 2, 2, 1], 'op2_hw': [8, 8], 'op2_strides': [1, 1, 1, 1],
                  'input_from_rec': alt_input,
                  'op1_pad': 'VALID', 'op2_pad': 'SAME',
                  'name': 'DC3', 'subdir': subdir, 'trainable': trainable, 'alt_load_subdir': alt_load_subdir}

    inv['DC2'] = {'classifier': 'alexnet', 'inv_input_name': 'lrn1:0', 'inv_target_name': 'conv1/lin:0',
                  'hidden_channels': 96, 'rec_name': 'c1l_rec',
                  'op1_hw': [8, 8], 'op1_strides': [1, 1, 1, 1], 'op2_hw': [8, 8], 'op2_strides': [1, 1, 1, 1],
                  'input_from_rec': alt_input,
                  'name': 'DC2', 'subdir': subdir, 'trainable': trainable, 'alt_load_subdir': alt_load_subdir}

    inv['DC1'] = {'classifier': 'alexnet', 'inv_input_name': 'conv1/lin:0', 'inv_target_name': 'rgb_scaled:0',
                  'hidden_channels': 96, 'rec_name': 'rgb_rec',
                  'op1_hw': [11, 11], 'op1_strides': [1, 4, 4, 1], 'op2_hw': [11, 11], 'op2_strides': [1, 1, 1, 1],
                  'input_from_rec': alt_input,
                  'op1_pad': 'VALID', 'name': 'DC1', 'subdir': subdir, 'trainable': trainable,
                  'alt_load_subdir': alt_load_subdir}

    return inv


def vgg16_specs(subdir='solotrain', alt_input=None, alt_load_subdir=None, trainable=True):
    inv = dict()
    
    inv['DC10'] = {'classifier': 'vgg16', 'inv_input_name': 'pool3:0', 'inv_target_name': 'conv3_3/lin:0',
                   'hidden_channels': 256, 'rec_name': 'c33l_rec',
                   'op1_hw': [5, 5], 'op1_strides': [1, 2, 2, 1], 'op2_hw': [5, 5], 'op2_strides': [1, 1, 1, 1],
                   'input_from_rec': alt_input,
                   'name': 'DC10', 'subdir': subdir, 'trainable': trainable, 'alt_load_subdir': alt_load_subdir}

    inv['DC9'] = {'classifier': 'vgg16', 'inv_input_name': 'conv3_3/lin:0', 'inv_target_name': 'conv3_2/lin:0',
                  'hidden_channels': 256, 'rec_name': 'c32l_rec',
                  'op1_hw': [5, 5], 'op1_strides': [1, 1, 1, 1], 'op2_hw': [5, 5], 'op2_strides': [1, 1, 1, 1],
                  'input_from_rec': alt_input,
                  'name': 'DC9', 'subdir': subdir, 'trainable': trainable, 'alt_load_subdir': alt_load_subdir}

    inv['DC8'] = {'classifier': 'vgg16', 'inv_input_name': 'conv3_2/lin:0', 'inv_target_name': 'conv3_1/lin:0',
                  'hidden_channels': 256, 'rec_name': 'c31l_rec',
                  'op1_hw': [5, 5], 'op1_strides': [1, 1, 1, 1], 'op2_hw': [5, 5], 'op2_strides': [1, 1, 1, 1],
                  'input_from_rec': alt_input,
                  'name': 'DC8', 'subdir': subdir, 'trainable': trainable, 'alt_load_subdir': alt_load_subdir}

    inv['DC7'] = {'classifier': 'vgg16', 'inv_input_name': 'conv3_1/lin:0', 'inv_target_name': 'pool2:0',
                  'hidden_channels': 256, 'rec_name': 'pool2_rec',
                  'op1_hw': [5, 5], 'op1_strides': [1, 1, 1, 1], 'op2_hw': [5, 5], 'op2_strides': [1, 1, 1, 1],
                  'input_from_rec': alt_input,
                  'name': 'DC7', 'subdir': subdir, 'trainable': trainable, 'alt_load_subdir': alt_load_subdir}
    
    inv['DC6'] = {'classifier': 'vgg16', 'inv_input_name': 'pool2:0', 'inv_target_name': 'conv2_2/lin:0',
                  'hidden_channels': 128, 'rec_name': 'c22l_rec',
                  'op1_hw': [5, 5], 'op1_strides': [1, 2, 2, 1], 'op2_hw': [5, 5], 'op2_strides': [1, 1, 1, 1],
                  'input_from_rec': alt_input,
                  'name': 'DC6', 'subdir': subdir, 'trainable': trainable, 'alt_load_subdir': alt_load_subdir}

    inv['DC5'] = {'classifier': 'vgg16', 'inv_input_name': 'conv2_2/lin:0', 'inv_target_name': 'conv2_1/lin:0',
                  'hidden_channels': 128, 'rec_name': 'c21l_rec',
                  'op1_hw': [5, 5], 'op1_strides': [1, 1, 1, 1], 'op2_hw': [5, 5], 'op2_strides': [1, 1, 1, 1],
                  'input_from_rec': alt_input,
                  'name': 'DC5', 'subdir': subdir, 'trainable': trainable, 'alt_load_subdir': alt_load_subdir}

    inv['DC4'] = {'classifier': 'vgg16', 'inv_input_name': 'conv2_1/lin:0', 'inv_target_name': 'pool1:0',
                  'hidden_channels': 128, 'rec_name': 'pool1_rec',
                  'op1_hw': [5, 5], 'op1_strides': [1, 1, 1, 1], 'op2_hw': [5, 5], 'op2_strides': [1, 1, 1, 1],
                  'input_from_rec': alt_input,
                  'name': 'DC4', 'subdir': subdir, 'trainable': trainable, 'alt_load_subdir': alt_load_subdir}

    inv['DC3'] = {'classifier': 'vgg16', 'inv_input_name': 'pool1:0', 'inv_target_name': 'conv1_2/lin:0',
                  'hidden_channels': 64, 'rec_name': 'c12l_rec',
                  'op1_hw': [5, 5], 'op1_strides': [1, 2, 2, 1], 'op2_hw': [5, 5], 'op2_strides': [1, 1, 1, 1],
                  'input_from_rec': alt_input,
                  'name': 'DC3', 'subdir': subdir, 'trainable': trainable, 'alt_load_subdir': alt_load_subdir}

    inv['DC2'] = {'classifier': 'vgg16', 'inv_input_name': 'conv1_2/lin:0', 'inv_target_name': 'conv1_1/lin:0',
                  'hidden_channels': 64, 'rec_name': 'c11l_rec',
                  'op1_hw': [5, 5], 'op1_strides': [1, 1, 1, 1], 'op2_hw': [5, 5], 'op2_strides': [1, 1, 1, 1],
                  'input_from_rec': alt_input,
                  'name': 'DC2', 'subdir': subdir, 'trainable': trainable, 'alt_load_subdir': alt_load_subdir}

    inv['DC1'] = {'classifier': 'vgg16', 'inv_input_name': 'conv1_1/lin:0', 'inv_target_name': 'rgb_scaled:0',
                  'hidden_channels': 64, 'rec_name': 'rgb_rec',
                  'op1_hw': [5, 5], 'op1_strides': [1, 1, 1, 1], 'op2_hw': [5, 5], 'op2_strides': [1, 1, 1, 1],
                  'input_from_rec': alt_input,
                  'name': 'DC1', 'subdir': subdir, 'trainable': trainable, 'alt_load_subdir': alt_load_subdir}

    return inv
