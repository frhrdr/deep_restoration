import tensorflow as tf
import numpy as np
from modules.foe_full_prior import FoEFullPrior
from utils.patch_prior_losses import logistic_ensemble_mrf_loss, student_ensemble_mrf_loss


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

    def build(self, scope_suffix='', featmap_tensor=None, masks=None):

        with tf.variable_scope(self.name):

            n_features_raw = self.ph * self.pw * self.n_channels
            whitening_tensor = tf.get_variable('whiten_mat', shape=[self.n_features_white, n_features_raw],
                                               dtype=tf.float32, trainable=False)

            ica_a, ica_w, _, _ = self.make_normed_filters(trainable=False, squeeze_alpha=True)

            whitened_mixing = tf.matmul(whitening_tensor, ica_w, transpose_a=True)

            if masks is None:
                whitened_mixing = self.add_dropout(whitened_mixing)  # DROPOUT

            featmap = featmap_tensor or self.get_tensors()

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

            if masks is not None:
                # xw shape [n_cliques, n_components] -> [ensemble_size, n_cliques, n_components]
                n_cliques = xw.get_shape()[0].value
                xw = tf.reshape(xw, shape=(1, n_cliques, self.n_components))
                ensemble_size = masks.get_shape()[0].value
                masks = tf.reshape(masks, shape=(ensemble_size, 1, self.n_components))
                xw = tf.transpose(tf.transpose(xw) @ tf.transpose(masks))
                print('should be [ensemble_size, n_cliques, n_components]:', xw.get_shape())

            self.loss = self.mrf_loss(xw, ica_a)
            self.var_list.append(whitening_tensor)

    def score_matching_graph(self, batch_size):
        ica_a, ica_w, extra_op, _ = self.make_normed_filters(trainable=True, squeeze_alpha=False)

        ica_w_drop = self.add_dropout(ica_w)  # DROPOUT

        x_pl = self.get_x_placeholder(batch_size)
        loss, term_1, term_2 = self.score_matching_loss(x_mat=x_pl, ica_w=ica_w_drop, ica_a=ica_a)
        return loss, term_1, term_2, x_pl, ica_a, ica_w, extra_op

    def add_dropout(self, weights, debug=False):
        weight_shape = [k.value for k in weights.get_shape()]
        assert len(weight_shape) == 2 and weight_shape[1] == self.n_components
        seed = 42
        dropped_out_weights = tf.nn.dropout(weights, self.dropout_prob, noise_shape=[1, self.n_components], seed=seed)

        if debug:
            dropout_vis = tf.greater(tf.abs(dropped_out_weights[0, :]), tf.constant(0.0))
            dropped_out_weights = tf.Print(dropped_out_weights, [dropout_vis], summarize=20)

        if self.make_switch:
            self.activate_dropout = tf.placeholder(dtype=tf.bool)
            return tf.cond(self.activate_dropout, lambda: dropped_out_weights, lambda: weights)
        elif self.activate_dropout:
            return dropped_out_weights
        else:
            return weights

    def mrf_loss(self, xw, ica_a_flat):
        if len(xw.get_shape()) == 2:
            return super().mrf_loss(xw, ica_a_flat)
        else:
            assert len(xw.get_shape()) == 3
            if self.dist == 'logistic':
                return logistic_ensemble_mrf_loss(xw, ica_a_flat)
            elif self.dist == 'student':
                return student_ensemble_mrf_loss(xw, ica_a_flat)
            else:
                raise ValueError

    def make_dropout_masks(self, ensemble_size, n_iterations):
        # every part of the ensemble needs one mask per iteration of size n_components -> shape [n_it, n_ens, n_comp]
        masks_size = (n_iterations, ensemble_size, self.n_components)
        mask_mat = np.random.randint(0, 2, size=masks_size).astype(np.float32)
        return tf.constant(mask_mat, name='dropout masks')

    def masked_forward_opt_adam(self, input_featmap, learning_rate, n_iterations, masks,
                                make_switch=False, beta1=0.9, beta2=0.999, eps=1e-8):

        def apply_adam(variable, gradients, m_acc, v_acc, iteration):
            beta1_tsr = tf.constant(beta1, dtype=tf.float32)
            beta2_tsr = tf.constant(beta2, dtype=tf.float32)
            m_new = beta1_tsr * m_acc + (1.0 - beta1_tsr) * gradients
            v_new = beta2_tsr * v_acc + (1.0 - beta2_tsr) * gradients ** 2
            m_hat = m_new / (1.0 - beta1_tsr ** iteration)
            v_hat = v_new / (1.0 - beta2_tsr ** iteration)
            variable -= learning_rate * m_hat / (tf.sqrt(v_hat) + eps)

            return variable, m_new, v_new

        def cond(*args):
            return tf.not_equal(args[0], tf.constant(n_iterations, dtype=tf.float32))

        ensemble_size = masks.get_shape()[1].value

        def body(count, featmaps, m_acc, v_acc):
            it_masks = masks[count, :, :]

            count += 1
            self.build(featmap_tensor=featmap, masks=it_masks)
            losses = tf.split(self.loss, self.loss.get_shape()[0].value)
            featmap_grads = [tf.gradients(ys=k, xs=featmap) for k in losses]
            featmap_grads = tf.stack(featmap_grads, axis=0)
            featmap, m_acc, v_acc = apply_adam(featmaps, featmap_grads, m_acc, v_acc, count)
            return count, featmap, m_acc, v_acc

        featmap_shape = [k.value for k in input_featmap.get_shape()]
        input_featmaps = tf.stack([input_featmap] * ensemble_size, axis=0)
        m_init = tf.constant(np.zeros([ensemble_size] + featmap_shape), dtype=tf.float32)
        v_init = tf.constant(np.zeros([ensemble_size] + featmap_shape), dtype=tf.float32)
        count_init = tf.constant(0, dtype=tf.float32)
        _, final_featmap, _, _ = tf.while_loop(cond=cond, body=body,
                                               loop_vars=[count_init, input_featmaps, m_init, v_init])

        return final_featmap
