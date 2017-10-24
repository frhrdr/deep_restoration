from modules.core_modules import Module
import tensorflow as tf

from modules.loss_modules import MSELoss


class SplitModule(Module):

    def __init__(self, name_to_split, img_slice_name, rec_slice_name, name=None):
        super().__init__(name_to_split, name=name)
        self.img_slice_name = img_slice_name
        self.rec_slice_name = rec_slice_name

    def build(self, scope_suffix=''):
        to_split = self.get_tensors()
        # assert to_split.get_shape()[0].value == 2
        if len(to_split.get_shape()) == 2:
            to_split = tf.reshape(to_split, shape=[2, -1, 1, 1])

        tf.slice(to_split, [0, 0, 0, 0], [1, -1, -1, -1], name=self.img_slice_name)
        tf.slice(to_split, [1, 0, 0, 0], [1, -1, -1, -1], name=self.rec_slice_name)


def lin_split_and_mse(layer, add_loss, mse_name=None):
    if layer < 6:
        long = 'conv{}/lin:0'.format(layer)
        short = 'c{}l'.format(layer)
    else:
        long = 'fc{}/lin:0'.format(layer)
        short = 'f{}l'.format(layer)
    img_slice = 'img_rep_' + short
    rec_slice = 'rec_rep_' + short
    mse_name = mse_name or 'MSE_' + short
    split = SplitModule(name_to_split=long, img_slice_name=img_slice,
                        rec_slice_name=rec_slice, name='Split{}'.format(layer))
    mse = MSELoss(target=img_slice + ':0', reconstruction=rec_slice + ':0',
                  name=mse_name)
    mse.add_loss = add_loss
    return split, mse
