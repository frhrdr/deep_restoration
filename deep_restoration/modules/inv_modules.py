import tensorflow as tf
from modules.core_modules import InversionModule


class ScaleConvConvModule(InversionModule):

    def __init__(self, inv_input_name, inv_target_name, hidden_channels, rec_name,
                 op1_hw, op1_strides, op2_hw, op2_strides, input_from_rec=None,
                 op1_pad='SAME', op2_pad='SAME',
                 name='ScaleConvConvModule', dir_name='scale_conv_conv_module', load_name='ScaleConvConvModule',
                 subdir='', trainable=False, alt_load_subdir=None):
        super().__init__(inv_input_name, inv_target_name, hidden_channels, rec_name,
                         op1_hw, op1_strides, op2_hw, op2_strides, input_from_rec=input_from_rec,
                         op1_pad=op1_pad, op2_pad=op2_pad, name=name, dir_name=dir_name, load_name=load_name,
                         subdir=subdir, trainable=trainable, alt_load_subdir=alt_load_subdir)

    def build(self, scope_suffix=''):
        in_tensor, inv_target = self.get_tensors()
        out_shape = [k.value for k in inv_target.get_shape()[1:]]
        with tf.variable_scope(self.name):

            out_h, out_w, out_c = out_shape
            scaled_input = tf.image.resize_images(in_tensor, size=[out_h, out_w],
                                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            batch_size, in_h, in_w, in_c = [k.value for k in in_tensor.get_shape()]

            conv_filter_1 = tf.get_variable('conv_filter_1', shape=[self.op1_height, self.op1_width,
                                                                    in_c, self.hidden_channels])

            conv = tf.nn.conv2d(scaled_input, filter=conv_filter_1, strides=self.op1_strides, padding=self.op1_pad)
            conv_bias_1 = tf.get_variable('conv_bias_1', shape=[self.hidden_channels])
            biased_conv = tf.nn.bias_add(conv, conv_bias_1)
            relu = tf.nn.relu(biased_conv)

            conv_filter_2 = tf.get_variable('conv_filter', shape=[self.op2_height, self.op2_width,
                                                                  self.hidden_channels, out_c])
            conv = tf.nn.conv2d(relu, filter=conv_filter_2, strides=self.op2_strides, padding=self.op2_pad)
            conv_bias_2 = tf.get_variable('conv_bias', shape=[out_c])
            tf.nn.bias_add(conv, conv_bias_2, name=self.rec_name)

            self.var_list = [conv_filter_1, conv_bias_1, conv_filter_2, conv_bias_2]


class ConvDeconvModule(InversionModule):

    def __init__(self, inv_input_name, inv_target_name, hidden_channels, rec_name,
                 op1_hw, op1_strides, op2_hw, op2_strides, input_from_rec=None,
                 op1_pad='SAME', op2_pad='SAME',
                 name='ConvDeconvModule', dir_name='conv_deconv_module', load_name='ConvDeconvModule',
                 subdir='', trainable=False, alt_load_subdir=None):
        super().__init__(inv_input_name, inv_target_name, hidden_channels, rec_name,
                         op1_hw, op1_strides, op2_hw, op2_strides, input_from_rec=input_from_rec,
                         op1_pad=op1_pad, op2_pad=op2_pad, name=name, dir_name=dir_name, load_name=load_name,
                         subdir=subdir, trainable=trainable, alt_load_subdir=alt_load_subdir)

    def build(self, scope_suffix=''):
        in_tensor, inv_target = self.get_tensors()
        out_shape = [k.value for k in inv_target.get_shape()[1:]]
        with tf.variable_scope(self.name):

            batch_size, in_h, in_w, in_c = [k.value for k in in_tensor.get_shape()]
            out_h, out_w, out_c = out_shape

            conv_filter = tf.get_variable('conv_filter', shape=[self.op1_height, self.op1_width,
                                                                in_c, self.hidden_channels])

            conv = tf.nn.conv2d(in_tensor, filter=conv_filter, strides=self.op1_strides, padding=self.op1_pad)
            conv_bias = tf.get_variable('conv_bias', shape=[self.hidden_channels])
            biased_conv = tf.nn.bias_add(conv, conv_bias)
            relu = tf.nn.relu(biased_conv)

            deconv_filter = tf.get_variable('deconv_filter', shape=[self.op2_height, self.op2_width,
                                                                    out_c, self.hidden_channels])
            deconv = tf.nn.conv2d_transpose(relu, filter=deconv_filter, output_shape=[batch_size, out_h, out_w, out_c],
                                            strides=self.op2_strides, padding=self.op2_pad)
            deconv_bias = tf.get_variable('deconv_bias', shape=[out_c])
            tf.nn.bias_add(deconv, deconv_bias, name=self.rec_name)

            self.var_list = [conv_filter, conv_bias, deconv_filter, deconv_bias]


class DeconvConvModule(InversionModule):

    def __init__(self, inv_input_name, inv_target_name, hidden_channels, rec_name,
                 op1_hw, op1_strides, op2_hw, op2_strides, input_from_rec=None,
                 hid_hw=None, hid_pad=None, op1_pad='SAME', op2_pad='SAME',
                 name='DeconvConvModule', dir_name='deconv_conv_module', load_name='DeconvConvModule',
                 subdir='', trainable=False, alt_load_subdir=None):
        super().__init__(inv_input_name, inv_target_name, hidden_channels, rec_name,
                         op1_hw, op1_strides, op2_hw, op2_strides, input_from_rec=input_from_rec,
                         op1_pad=op1_pad, op2_pad=op2_pad, name=name, dir_name=dir_name, load_name=load_name,
                         subdir=subdir, trainable=trainable, alt_load_subdir=alt_load_subdir)
        self.hid_hw = hid_hw
        self.hid_pad = hid_pad

    def build(self, scope_suffix=''):
        in_tensor, inv_target = self.get_tensors()
        out_shape = [k.value for k in inv_target.get_shape()[1:]]
        with tf.variable_scope(self.name):

            batch_size, in_h, in_w, in_c = [k.value for k in in_tensor.get_shape()]
            out_h, out_w, out_c = out_shape

            hid_h, hid_w = (out_h, out_w) if self.hid_hw is None else self.hid_hw

            # assume for now that the hidden layer has the same dims as the output
            deconv_filter = tf.get_variable('deconv_filter', shape=[self.op1_height, self.op1_width,
                                                                    self.hidden_channels, in_c])
            deconv = tf.nn.conv2d_transpose(in_tensor, filter=deconv_filter,
                                            output_shape=[batch_size, hid_h, hid_w, self.hidden_channels],
                                            strides=self.op1_strides, padding=self.op1_pad)
            deconv_bias = tf.get_variable('deconv_bias', shape=[self.hidden_channels])
            biased_conv = tf.nn.bias_add(deconv, deconv_bias)
            relu = tf.nn.relu(biased_conv)
            if self.hid_pad is not None:
                relu = tf.pad(relu, self.hid_pad, mode='REFLECT')
            conv_filter = tf.get_variable('conv_filter', shape=[self.op2_height, self.op2_width,
                                                                self.hidden_channels, out_c])
            conv = tf.nn.conv2d(relu, filter=conv_filter, strides=self.op2_strides, padding=self.op2_pad)
            conv_bias = tf.get_variable('conv_bias', shape=[out_c])
            tf.nn.bias_add(conv, conv_bias, name=self.rec_name)

            self.var_list = [conv_filter, conv_bias, deconv_filter, deconv_bias]


class DeconvDeconvModule(InversionModule):

    def __init__(self, inv_input_name, inv_target_name, hidden_channels, rec_name,
                 op1_hw, op1_strides, op2_hw, op2_strides, input_from_rec=None,
                 op1_pad='SAME', op2_pad='SAME',
                 name='DeconvDeconvModule', dir_name='deconv_deconv_module', load_name='DeconvDeconvModule',
                 subdir='', trainable=False, alt_load_subdir=None):
        super().__init__(inv_input_name, inv_target_name, hidden_channels, rec_name,
                         op1_hw, op1_strides, op2_hw, op2_strides, input_from_rec=input_from_rec,
                         op1_pad=op1_pad, op2_pad=op2_pad, name=name, dir_name=dir_name, load_name=load_name,
                         subdir=subdir, trainable=trainable, alt_load_subdir=alt_load_subdir)

    def build(self, scope_suffix=''):
        in_tensor, inv_target = self.get_tensors()
        out_shape = [k.value for k in inv_target.get_shape()[1:]]
        with tf.variable_scope(self.name):

            batch_size, in_h, in_w, in_c = [k.value for k in in_tensor.get_shape()]
            out_h, out_w, out_c = out_shape

            deconv_filter_1 = tf.get_variable('deconv_filter_1',
                                              shape=[self.op1_height, self.op1_width,
                                                     self.hidden_channels, in_c])
            deconv_1 = tf.nn.conv2d_transpose(in_tensor, filter=deconv_filter_1,
                                              output_shape=[batch_size, out_h,
                                                            out_w, self.hidden_channels],
                                              strides=self.op1_strides, padding=self.op1_pad)
            deconv_bias_1 = tf.get_variable('deconv_bias', shape=[self.hidden_channels])
            biased_deconv_1 = tf.nn.bias_add(deconv_1, deconv_bias_1)

            relu = tf.nn.relu(biased_deconv_1)

            deconv_filter_2 = tf.get_variable('deconv_filter_2',
                                              shape=[self.op2_height, self.op2_width,
                                                     out_c, self.hidden_channels])
            deconv_2 = tf.nn.conv2d_transpose(relu, filter=deconv_filter_2,
                                              output_shape=[batch_size, out_h,
                                                            out_w, out_c],
                                              strides=self.op2_strides, padding=self.op2_pad)
            deconv_bias_2 = tf.get_variable('deconv_bias_2', shape=[out_c])
            tf.nn.bias_add(deconv_2, deconv_bias_2, name=self.rec_name)

            self.var_list = [deconv_filter_1, deconv_bias_1, deconv_filter_2, deconv_bias_2]
