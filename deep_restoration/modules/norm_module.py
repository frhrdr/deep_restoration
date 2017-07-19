from modules.loss_modules import Module
import tensorflow as tf


class NormModule(Module):

    def __init__(self, name_to_norm, out_name, offset=None, scale=None):
        super().__init__(name_to_norm)
        self.offset = offset
        self.scale = scale
        self.out_name = out_name

    def build(self, scope_suffix=''):
        to_norm = self.get_in_tensors()
        offset = self.offset if self.offset else tf.reduce_mean(to_norm)
        scale = self.scale if self.scale else 1 / tf.norm(to_norm)
        tf.multiply(to_norm - offset, scale, name=self.out_name)
