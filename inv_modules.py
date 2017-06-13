import tensorflow as tf
import os
from loss_modules import Module

# spec1 = dict(op1_height=5, op1_width=5, op1_strides=[1, 2, 2, 1], op1_pad='SAME',
#              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1], op2_pad='SAME',
#              hidden_channels=256, target_shape=[28, 28, 256],
#              inv_input_name='pool3:0', inv_target_name='conv3_3/relu:0',
#              rec_name='conv3_3_rec')


class TrainedModule(Module):

    def __init__(self, tensor_names, name, load_path, trainable):
        super().__init__(tensor_names)
        self.name = name
        self.load_path = load_path
        self.trainable = trainable
        self.var_list = []

    def load_weights(self, session):
        loader = tf.train.Saver(var_list=self.var_list)
        loader.restore(session, self.load_path)

    def save_weights(self, session, step):
        saver = tf.train.Saver(var_list=self.var_list)
        checkpoint_file = os.path.join(self.load_path, 'ckpt')
        saver.save(session, checkpoint_file, global_step=step, write_meta_graph=False)

    def build(self, scope_suffix):
        raise NotImplementedError

    def is_trainable(self):
        return self.trainable


class ConvDeconvModule(TrainedModule):

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

            self.var_list = [conv_filter, conv_bias, deconv_filter, deconv_bias]


class DeconvConvModule(TrainedModule):

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

            self.var_list = [conv_filter, conv_bias, deconv_filter, deconv_bias]


class DeconvDeconvModule(TrainedModule):

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

            self.var_list = [deconv_filter_1, deconv_bias_1, deconv_filter_2, deconv_bias_2]


class ScaleConvConvModule(TrainedModule):

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

            conv_filter_1 = tf.get_variable('conv_filter_1', shape=[self.specs['op1_height'], self.specs['op1_width'],
                                                                    in_c, self.specs['hidden_channels']])

            conv = tf.nn.conv2d(scaled_input, filter=conv_filter_1, strides=self.specs['op1_strides'], padding=op1_pad)
            conv_bias_1 = tf.get_variable('conv_bias_1', shape=[self.specs['hidden_channels']])
            biased_conv = tf.nn.bias_add(conv, conv_bias_1)
            relu = tf.nn.relu(biased_conv)

            conv_filter_2 = tf.get_variable('conv_filter', shape=[self.specs['op2_height'], self.specs['op2_width'],
                                                                  self.specs['hidden_channels'], out_c])
            conv = tf.nn.conv2d(relu, filter=conv_filter_2, strides=self.specs['op2_strides'], padding=op2_pad)
            conv_bias_2 = tf.get_variable('conv_bias', shape=[out_c])
            tf.nn.bias_add(conv, conv_bias_2, name=self.specs['rec_name'])

            self.var_list = [conv_filter_1, conv_bias_1, conv_filter_2, conv_bias_2]