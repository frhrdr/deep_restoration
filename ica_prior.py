import tensorflow as tf
import numpy as np
from loss_modules import LearnedPriorLoss


class ICAPrior(LearnedPriorLoss):

    def __init__(self, tensor_names, weighting, name, load_path, trainable, filter_dims, input_scaling):
        super().__init__(tensor_names, weighting, name, load_path, trainable)
        self.filter_dims = filter_dims
        self.input_scaling = input_scaling

    @staticmethod
    def flattening_filter(dims):
        assert len(dims) == 3
        f = np.zeros((dims[0], dims[1], dims[2], dims[0] * dims[1] * dims[2]))
        for idx in range(f.shape[3]):
            x = (idx // (dims[1] * dims[2])) % dims[0]
            y = (idx // dims[2]) % dims[1]
            z = idx % dims[2]

            f[x, y, z, idx] = 1
        return f

    def build(self):
        tensor = self.get_tensors()
        dims = [s.value for s in tensor.get_shape()]
        assert len(dims) == 4
        filter_mat = self.flattening_filter((self.filter_dims[0], self.filter_dims[1], dims[3]))
        flat_filter = tf.constant(filter_mat, dtype=tf.float32)
        x_pad = (dims[1] - 1) / 2
        y_pad = (dims[2] - 1) / 2
        conv_input = tf.pad(tensor, paddings=[(0, 0), (x_pad, x_pad), (y_pad, y_pad), (0, 0)], mode='REFLECT')
        flat_patches = tf.nn.conv2d(conv_input, flat_filter, strides=[1, 1, 1, 1], padding='VALID')
        scaled_patches = flat_patches * self.input_scaling
        centered_patches = flat_patches - tf.stack([tf.reduce_mean(scaled_patches, axis=3)] * filter_mat.shape[3],
                                                   axis=3)
        whitening_tensor = tf.get_variable('whitening_mat', shape=[filter_mat.shape[3], filter_mat.shape[3] - 1],
                                           dtype=tf.float32, trainable=self.trainable)

        whitened_patches = centered_patches @ whitening_tensor

        ica_a = tf.get_variable('ica_a', shape=[])
        ica_w = tf.get_variable('ica_w', shape=[])





