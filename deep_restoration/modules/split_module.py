from modules.loss_modules import Module
import tensorflow as tf


class SplitModule(Module):

    def __init__(self, name_to_split, img_slice_name, rec_slice_name, name=None):
        super().__init__(name_to_split, name=name)
        self.img_slice_name = img_slice_name
        self.rec_slice_name = rec_slice_name

    def build(self, scope_suffix=''):
        to_split = self.get_in_tensors()
        # assert to_split.get_shape()[0].value == 2
        if len(to_split.get_shape()) == 2:
            to_split = tf.reshape(to_split, shape=[2, -1, 1, 1])

        tf.slice(to_split, [0, 0, 0, 0], [1, -1, -1, -1], name=self.img_slice_name)
        tf.slice(to_split, [1, 0, 0, 0], [1, -1, -1, -1], name=self.rec_slice_name)
