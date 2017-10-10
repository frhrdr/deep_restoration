import tensorflow as tf
from modules.foe_full_prior import FoEFullPrior


class FoEDropoutPrior(FoEFullPrior):
    def __init__(self, tensor_names, weighting, classifier, filter_dims, input_scaling,
                 n_components, n_channels, n_features_white,
                 dist='student', mean_mode='gc', sdev_mode='gc', whiten_mode='pca',
                 name=None, load_name=None, dir_name=None, load_tensor_names=None,
                 activate_dropout=True, make_switch=False, dropout_prob=0.5):
        super().__init__(tensor_names, weighting, classifier, filter_dims, input_scaling,
                         n_components, n_channels, n_features_white,
                         dist=dist, mean_mode=mean_mode, sdev_mode=sdev_mode, whiten_mode=whiten_mode,
                         name=name, load_name=load_name, dir_name=dir_name, load_tensor_names=load_tensor_names)
        self.activate_dropout = activate_dropout
        self.make_switch = make_switch
        self.dropout_prob = dropout_prob

    @staticmethod
    def assign_names(dist, name, load_name, dir_name, load_tensor_names, tensor_names):
        dist_options = ('student', 'logistic')
        assert dist in dist_options

        student_names = ('FoEStudentDropoutPrior', 'FoEStudentDropoutPrior', 'student_dropout_prior')
        logistic_names = ('FoELogisticDropoutPrior', 'FoELogisticDropoutPrior', 'logistic_dropout_prior')
        dist_names = student_names if dist == 'student' else logistic_names
        name = name if name is not None else dist_names[0]
        load_name = load_name if load_name is not None else dist_names[1]
        dir_name = dir_name if dir_name is not None else dist_names[2]
        load_tensor_names = load_tensor_names if load_tensor_names is not None else tensor_names

        return name, load_name, dir_name, load_tensor_names

    def build(self, scope_suffix='', featmap_tensor=None):

        with tf.variable_scope(self.name):

            n_features_raw = self.ph * self.pw * self.n_channels
            whitening_tensor = tf.get_variable('whiten_mat', shape=[self.n_features_white, n_features_raw],
                                               dtype=tf.float32, trainable=False)

            ica_a, ica_w, _, _ = self.make_normed_filters(trainable=False, squeeze_alpha=True)

            whitened_mixing = tf.matmul(whitening_tensor, ica_w, transpose_a=True)

            whitened_mixing = self.add_dropout(whitened_mixing)  # DROPOUT

            featmap = self.get_tensors() if featmap_tensor is None else featmap_tensor

            if self.mean_mode in ('gc', 'gf') and self.sdev_mode in ('gc', 'gf'):
                normed_featmap = self.norm_feat_map_directly(featmap)  # shape [1, h, w, n_channels]

                whitened_mixing = tf.reshape(whitened_mixing, shape=[self.n_channels, self.ph,
                                                                     self.pw, self.n_components])
                whitened_mixing = tf.transpose(whitened_mixing, perm=[1, 2, 0, 3])

                xw = tf.nn.conv2d(normed_featmap, whitened_mixing, strides=[1, 1, 1, 1], padding='VALID')
                xw = tf.reshape(xw, shape=[-1, self.n_components])

            else:
                normed_patches = self.shape_and_norm_featmap(featmap)
                n_patches = normed_patches.get_shape()[0].value
                normed_patches = tf.reshape(normed_patches, shape=[n_patches, n_features_raw])
                xw = tf.matmul(normed_patches, whitened_mixing)

            self.loss = self.mrf_loss(xw, ica_a)
            self.var_list.append(whitening_tensor)

    def score_matching_graph(self, batch_size):
        ica_a, ica_w, extra_op, _ = self.make_normed_filters(trainable=True, squeeze_alpha=False)

        ica_w_drop = self.add_dropout(ica_w)  # DROPOUT

        x_pl = self.get_x_placeholder(batch_size)
        loss, term_1, term_2 = self.score_matching_loss(x_mat=x_pl, ica_w=ica_w_drop, ica_a=ica_a)
        return loss, term_1, term_2, x_pl, ica_a, ica_w, extra_op

    def add_dropout(self, weights):
        weight_shape = [k.value for k in weights.get_shape()]
        assert len(weight_shape) == 2 and weight_shape[1] == self.n_components
        seed = 42
        dropped_out_weights = tf.nn.dropout(weights, self.dropout_prob, noise_shape=[1, self.n_components], seed=seed)

        if self.make_switch:
            self.activate_dropout = tf.placeholder(dtype=tf.bool)
            return tf.cond(self.activate_dropout, lambda: dropped_out_weights, lambda: weights)
        elif self.activate_dropout:
            return dropped_out_weights
        else:
            return weights
