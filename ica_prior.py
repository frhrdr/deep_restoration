import tensorflow as tf
import numpy as np
from loss_modules import LearnedPriorLoss


class ICAPrior(LearnedPriorLoss):

    def __init__(self, tensor_names, weighting, name, load_path, trainable, filter_dims, input_scaling, n_components):
        super().__init__(tensor_names, weighting, name, load_path, trainable)
        self.filter_dims = filter_dims  #
        self.input_scaling = input_scaling
        self.n_components = n_components

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
        with tf.variable_scope(self.name):
            tensor = self.get_tensors()
            dims = [s.value for s in tensor.get_shape()]
            assert len(dims) == 4
            filter_mat = self.flattening_filter((self.filter_dims[0], self.filter_dims[1], dims[3]))
            flat_filter = tf.constant(filter_mat, dtype=tf.float32)
            x_pad = ((dims[1] - 1) // 2, int(np.ceil((dims[1] - 1) / 2)))
            y_pad = ((dims[2] - 1) // 2, int(np.ceil((dims[2] - 1) / 2)))
            conv_input = tf.pad(tensor, paddings=[(0, 0), x_pad, y_pad, (0, 0)], mode='REFLECT')
            flat_patches = tf.nn.conv2d(conv_input, flat_filter, strides=[1, 4, 4, 1], padding='VALID')
            scaled_patches = flat_patches * self.input_scaling
            centered_patches = flat_patches - tf.stack([tf.reduce_mean(scaled_patches, axis=3)] * filter_mat.shape[3],
                                                       axis=3)

            n_features_raw = filter_mat.shape[3]
            n_features_white = n_features_raw - 1

            whitening_tensor = tf.get_variable('whiten_mat', shape=[n_features_white, n_features_raw],
                                               dtype=tf.float32, trainable=self.trainable)

            centered_patches = tf.reshape(centered_patches, shape=[-1, n_features_raw])

            ica_a = tf.get_variable('ica_a', shape=[self.n_components, 1], trainable=self.trainable, dtype=tf.float32)
            ica_w = tf.get_variable('ica_w', shape=[n_features_white, self.n_components],
                                    trainable=self.trainable, dtype=tf.float32)
            ica_a_squeezed = tf.squeeze(ica_a)

            whitened_mixing = tf.matmul(whitening_tensor, ica_w, transpose_a=True)
            xw = tf.matmul(centered_patches, whitened_mixing)
            neg_g_wx = tf.log(0.5 * (tf.exp(-xw) + tf.exp(xw))) * ica_a_squeezed
            neg_log_p_patches = tf.reduce_sum(neg_g_wx, axis=1)
            naive_mean = tf.reduce_mean(neg_log_p_patches, name='loss')
            self.loss = naive_mean

            self.var_list = [ica_a, ica_w, whitening_tensor]
