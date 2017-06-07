import tensorflow as tf

class InversionModule:

    def __init__(self, specs):
        self.specs = specs

    def build(self, scope_suffix):
        raise NotImplementedError


class ConvDeconvModule(InversionModule):

    def build(self, scope_suffix):
        graph = tf.get_default_graph()
        in_tensor = graph.get_tensor_by_name(self.specs['inv_input_name'])
        inv_target = graph.get_tensor_by_name(self.specs['inv_target_name'])
        out_shape = [k.value for k in inv_target.get_shape()[1:]]
        with tf.variable_scope('module_' + scope_suffix):
            if 'op1_pad' in self.specs:
                op1_pad = self.specs['op1_pad']
            else:
                op1_pad = 'SAME'

            if 'op2_pad' in self.specs:
                op2_pad = self.specs['op2_pad']
            else:
                op2_pad = 'SAME'

            batch_size, in_h, in_w, in_c = [k.value for k in in_tensor.get_shape()]
            out_h, out_w, out_c = out_shape

            conv_filter = tf.get_variable('conv_filter', shape=[self.specs['op1_height'], self.specs['op1_width'],
                                                                in_c, self.specs['hidden_channels']])

            conv = tf.nn.conv2d(in_tensor, filter=conv_filter, strides=self.specs['op1_strides'], padding=op1_pad)
            conv_bias = tf.get_variable('conv_bias', shape=[self.specs['hidden_channels']])
            biased_conv = tf.nn.bias_add(conv, conv_bias)
            relu = tf.nn.relu(biased_conv)

            deconv_filter = tf.get_variable('deconv_filter', shape=[self.specs['op2_height'], self.specs['op2_width'],
                                                                    out_c, self.specs['hidden_channels']])
            deconv = tf.nn.conv2d_transpose(relu, filter=deconv_filter, output_shape=[batch_size, out_h, out_w, out_c],
                                            strides=self.specs['op2_strides'], padding=op2_pad)
            deconv_bias = tf.get_variable('deconv_bias', shape=[out_c])
            tf.nn.bias_add(deconv, deconv_bias, name=self.specs['rec_name'])


class DeconvConvModule(InversionModule):

    def build(self, scope_suffix):
        graph = tf.get_default_graph()
        in_tensor = graph.get_tensor_by_name(self.specs['inv_input_name'])
        inv_target = graph.get_tensor_by_name(self.specs['inv_target_name'])
        out_shape = [k.value for k in inv_target.get_shape()[1:]]
        with tf.variable_scope('module_' + scope_suffix):
            if 'op1_pad' in self.specs:
                op1_pad = self.specs['op1_pad']
            else:
                op1_pad = 'SAME'

            if 'op2_pad' in self.specs:
                op2_pad = self.specs['op2_pad']
            else:
                op2_pad = 'SAME'

            batch_size, in_h, in_w, in_c = [k.value for k in in_tensor.get_shape()]
            out_h, out_w, out_c = out_shape
            # assume for now that the hidden layer has the same dims as the output
            deconv_filter = tf.get_variable('deconv_filter', shape=[self.specs['op1_height'], self.specs['op1_width'],
                                                                    self.specs['hidden_channels'], in_c])
            deconv = tf.nn.conv2d_transpose(in_tensor, filter=deconv_filter,
                                            output_shape=[batch_size, out_h, out_w, self.specs['hidden_channels']],
                                            strides=self.specs['op1_strides'], padding=op1_pad)
            deconv_bias = tf.get_variable('deconv_bias', shape=[self.specs['hidden_channels']])
            biased_conv = tf.nn.bias_add(deconv, deconv_bias)
            relu = tf.nn.relu(biased_conv)

            conv_filter = tf.get_variable('conv_filter', shape=[self.specs['op2_height'], self.specs['op2_width'],
                                                                self.specs['hidden_channels'], out_c])
            conv = tf.nn.conv2d(relu, filter=conv_filter, strides=self.specs['op2_strides'], padding=op2_pad)
            conv_bias = tf.get_variable('conv_bias', shape=[out_c])
            tf.nn.bias_add(conv, conv_bias, name=self.specs['rec_name'])


class DeconvDeconvModule(InversionModule):

    def build(self, scope_suffix):
        graph = tf.get_default_graph()
        in_tensor = graph.get_tensor_by_name(self.specs['inv_input_name'])
        inv_target = graph.get_tensor_by_name(self.specs['inv_target_name'])
        out_shape = [k.value for k in inv_target.get_shape()[1:]]
        with tf.variable_scope('module_' + scope_suffix):
            if 'op1_pad' in self.specs:
                op1_pad = self.specs['op1_pad']
            else:
                op1_pad = 'SAME'

            if 'op2_pad' in self.specs:
                op2_pad = self.specs['op2_pad']
            else:
                op2_pad = 'SAME'

            batch_size, in_h, in_w, in_c = [k.value for k in in_tensor.get_shape()]
            out_h, out_w, out_c = out_shape

            deconv_filter_1 = tf.get_variable('deconv_filter_1',
                                              shape=[self.specs['op1_height'], self.specs['op1_width'],
                                                     self.specs['hidden_channels'], in_c])
            deconv_1 = tf.nn.conv2d_transpose(in_tensor, filter=deconv_filter_1,
                                              output_shape=[batch_size, out_h,
                                                            out_w, self.specs['hidden_channels']],
                                              strides=self.specs['op1_strides'], padding=op1_pad)
            deconv_bias_1 = tf.get_variable('deconv_bias', shape=[self.specs['hidden_channels']])
            biased_deconv_1 = tf.nn.bias_add(deconv_1, deconv_bias_1)

            relu = tf.nn.relu(biased_deconv_1)

            deconv_filter_2 = tf.get_variable('deconv_filter_2',
                                              shape=[self.specs['op2_height'], self.specs['op2_width'],
                                                     out_c, self.specs['hidden_channels']])
            deconv_2 = tf.nn.conv2d_transpose(relu, filter=deconv_filter_2,
                                              output_shape=[batch_size, out_h,
                                                            out_w, out_c],
                                              strides=self.specs['op2_strides'], padding=op2_pad)
            deconv_bias_2 = tf.get_variable('deconv_bias_2', shape=[out_c])
            tf.nn.bias_add(deconv_2, deconv_bias_2, name=self.specs['rec_name'])


class ScaleConvConvModule(InversionModule):

    def build(self, scope_suffix):
        graph = tf.get_default_graph()
        in_tensor = graph.get_tensor_by_name(self.specs['inv_input_name'])
        inv_target = graph.get_tensor_by_name(self.specs['inv_target_name'])
        out_shape = [k.value for k in inv_target.get_shape()[1:]]
        with tf.variable_scope('module_' + scope_suffix):
            if 'op1_pad' in self.specs:
                op1_pad = self.specs['op1_pad']
            else:
                op1_pad = 'SAME'

            if 'op2_pad' in self.specs:
                op2_pad = self.specs['op2_pad']
            else:
                op2_pad = 'SAME'

            out_h, out_w, out_c = out_shape
            scaled_input = tf.image.resize_images(in_tensor, size=[out_h, out_w],
                                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            batch_size, in_h, in_w, in_c = [k.value for k in in_tensor.get_shape()]

            conv_filter = tf.get_variable('conv_filter_1', shape=[self.specs['op1_height'], self.specs['op1_width'],
                                                                  in_c, self.specs['hidden_channels']])

            conv = tf.nn.conv2d(scaled_input, filter=conv_filter, strides=self.specs['op1_strides'], padding=op1_pad)
            conv_bias = tf.get_variable('conv_bias_1', shape=[self.specs['hidden_channels']])
            biased_conv = tf.nn.bias_add(conv, conv_bias)
            relu = tf.nn.relu(biased_conv)

            conv_filter = tf.get_variable('conv_filter', shape=[self.specs['op2_height'], self.specs['op2_width'],
                                                                self.specs['hidden_channels'], out_c])
            conv = tf.nn.conv2d(relu, filter=conv_filter, strides=self.specs['op2_strides'], padding=op2_pad)
            conv_bias = tf.get_variable('conv_bias', shape=[out_c])
            tf.nn.bias_add(conv, conv_bias, name=self.specs['rec_name'])