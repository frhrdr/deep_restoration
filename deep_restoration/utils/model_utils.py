import tensorflow as tf
# longer term -> use inv_modules


def conv_deconv_model(in_tensor, specs, out_shape):
    if 'op1_pad' in specs:
        op1_pad = specs['op1_pad']
    else:
        op1_pad = 'SAME'

    if 'op2_pad' in specs:
        op2_pad = specs['op2_pad']
    else:
        op2_pad = 'SAME'

    batch_size, in_h, in_w, in_c = [k.value for k in in_tensor.get_shape()]
    out_h, out_w, out_c = out_shape

    conv_filter = tf.get_variable('conv_filter', shape=[specs['op1_height'], specs['op1_width'],
                                                        in_c, specs['hidden_channels']])

    conv = tf.nn.conv2d(in_tensor, filter=conv_filter, strides=specs['op1_strides'], padding=op1_pad)

    conv_bias = tf.get_variable('conv_bias', shape=[specs['hidden_channels']])
    biased_conv = tf.nn.bias_add(conv, conv_bias)

    relu = tf.nn.relu(biased_conv)

    deconv_filter = tf.get_variable('deconv_filter',
                                    shape=[specs['op2_height'], specs['op2_width'],
                                           out_c, specs['hidden_channels']])
    deconv = tf.nn.conv2d_transpose(relu, filter=deconv_filter,
                                    output_shape=[batch_size, out_h,
                                                  out_w, out_c],
                                    strides=specs['op2_strides'], padding=op2_pad)

    deconv_bias = tf.get_variable('deconv_bias', shape=[out_c])
    reconstruction = tf.nn.bias_add(deconv, deconv_bias)
    return reconstruction


def deconv_conv_model(in_tensor, specs, out_shape):
    if 'op1_pad' in specs:
        op1_pad = specs['op1_pad']
    else:
        op1_pad = 'SAME'

    if 'op2_pad' in specs:
        op2_pad = specs['op2_pad']
    else:
        op2_pad = 'SAME'

    batch_size, in_h, in_w, in_c = [k.value for k in in_tensor.get_shape()]
    out_h, out_w, out_c = out_shape
    # assume for now that the hidden layer has the same dims as the output
    deconv_filter = tf.get_variable('deconv_filter',
                                    shape=[specs['op1_height'], specs['op1_width'],
                                           specs['hidden_channels'], in_c])
    deconv = tf.nn.conv2d_transpose(in_tensor, filter=deconv_filter,
                                    output_shape=[batch_size, out_h,
                                                  out_w, specs['hidden_channels']],
                                    strides=specs['op1_strides'], padding=op1_pad)
    deconv_bias = tf.get_variable('deconv_bias', shape=[specs['hidden_channels']])
    biased_conv = tf.nn.bias_add(deconv, deconv_bias)

    relu = tf.nn.relu(biased_conv)

    conv_filter = tf.get_variable('conv_filter', shape=[specs['op2_height'], specs['op2_width'],
                                                        specs['hidden_channels'], out_c])

    conv = tf.nn.conv2d(relu, filter=conv_filter, strides=specs['op2_strides'], padding=op2_pad)

    conv_bias = tf.get_variable('conv_bias', shape=[out_c])
    reconstruction = tf.nn.bias_add(conv, conv_bias)
    return reconstruction


def deconv_deconv_model(in_tensor, specs, out_shape):
    if 'op1_pad' in specs:
        op1_pad = specs['op1_pad']
    else:
        op1_pad = 'SAME'

    if 'op2_pad' in specs:
        op2_pad = specs['op2_pad']
    else:
        op2_pad = 'SAME'

    batch_size, in_h, in_w, in_c = [k.value for k in in_tensor.get_shape()]
    out_h, out_w, out_c = out_shape

    deconv_filter_1 = tf.get_variable('deconv_filter_1',
                                      shape=[specs['op1_height'], specs['op1_width'],
                                             specs['hidden_channels'], in_c])
    deconv_1 = tf.nn.conv2d_transpose(in_tensor, filter=deconv_filter_1,
                                      output_shape=[batch_size, out_h,
                                                    out_w, specs['hidden_channels']],
                                      strides=specs['op1_strides'], padding=op1_pad)
    deconv_bias_1 = tf.get_variable('deconv_bias', shape=[specs['hidden_channels']])
    biased_deconv_1 = tf.nn.bias_add(deconv_1, deconv_bias_1)

    relu = tf.nn.relu(biased_deconv_1)

    deconv_filter_2 = tf.get_variable('deconv_filter_2',
                                      shape=[specs['op2_height'], specs['op2_width'],
                                             out_c, specs['hidden_channels']])
    deconv_2 = tf.nn.conv2d_transpose(relu, filter=deconv_filter_2,
                                      output_shape=[batch_size, out_h,
                                                    out_w, out_c],
                                      strides=specs['op2_strides'], padding=op2_pad)
    deconv_bias_2 = tf.get_variable('deconv_bias_2', shape=[out_c])
    reconstruction = tf.nn.bias_add(deconv_2, deconv_bias_2)
    return reconstruction
