import numpy as np
import tensorflow as tf
from tf_vgg.vgg19 import Vgg19
from modules.core_modules import LossModule
from utils.feature_statistics import gram_tensor
from utils.filehandling import load_image


class MSELoss(LossModule):

    def __init__(self, target, reconstruction, weighting=1.0, name=None):
        super().__init__((target, reconstruction), weighting, name=name)

    def build(self, scope_suffix=''):
        with tf.variable_scope(self.name):
            tgt, rec = self.get_tensors()
            self.loss = tf.losses.mean_squared_error(tgt, rec)


class NormedMSELoss(MSELoss):

    def build(self, scope_suffix=''):
        with tf.variable_scope(self.name):
            tgt, rec = self.get_tensors()
            self.loss = tf.reduce_sum((tgt - rec) ** 2) / tf.reduce_sum(tgt * rec)


class SoftRangeLoss(LossModule):

    def __init__(self, tensor, alpha, weighting):
        super().__init__(tensor, weighting)
        self.alpha = alpha

    def build(self, scope_suffix=''):
        tensor = self.get_tensors()
        tmp = tf.reduce_sum((tensor - tf.reduce_mean(tensor, axis=[0, 1, 2, 3])) ** 2, axis=[3]) ** (self.alpha / 2)
        self.loss = tf.reduce_sum(tmp)


class TotalVariationLoss(LossModule):

    def __init__(self, tensor, beta, weighting):
        super().__init__(tensor, weighting)
        self.beta = beta

    def build(self, scope_suffix=''):
        tensor = self.get_tensors()

        h0 = tf.slice(tensor, [0, 0, 0, 0], [-1, tensor.get_shape()[1].value - 1, -1, -1])
        h1 = tf.slice(tensor, [0, 1, 0, 0], [-1, -1, -1, -1])

        v0 = tf.slice(tensor, [0, 0, 0, 0], [-1, -1, tensor.get_shape()[1].value - 1, -1])
        v1 = tf.slice(tensor, [0, 0, 1, 0], [-1, -1, -1, -1])

        h_diff = h0 - h1
        v_diff = v0 - v1

        d_sum = tf.pad(h_diff * h_diff, [[0, 0], [0, 1], [0, 0], [0, 0]]) + \
                tf.pad(v_diff * v_diff, [[0, 0], [0, 0], [0, 1], [0, 0]])

        self.loss = tf.reduce_sum(d_sum ** (self.beta / 2))


class VggScoreLoss(LossModule):

    def __init__(self, in_tensor_names, weighting=1.0, name=None, input_scaling=1.0):
        super().__init__(in_tensor_names, weighting, name=name)
        self.input_scaling = input_scaling

    def build(self, scope_suffix=''):
        with tf.variable_scope(self.name):
            tgt, rec = self.get_tensors()
            print(tgt, rec)
            assert tuple(k.value for k in tgt.get_shape()) == (1, 224, 224, 3)
            assert tuple(k.value for k in rec.get_shape()) == (1, 224, 224, 3)
            t_in = tf.concat((tgt, rec), axis=0)

            vgg = Vgg19()
            vgg.build(t_in, pool_mode='avg')

            graph = tf.get_default_graph()
            layers = ['{}/conv{}_1/relu:0'.format(self.name, k) for k in range(1, 6)]
            featmaps = [graph.get_tensor_by_name(n) for n in layers]
            featmap_splits = [tf.split(k, 2) for k in featmaps]
            print(featmap_splits)

            loss_acc = 0
            for split in featmap_splits:
                tgt, rec = split

                tgt_gram = gram_tensor(tf.squeeze(tgt))
                rec_gram = gram_tensor(tf.squeeze(rec))
                print(tgt_gram.get_shape())
                cost = tf.reduce_sum((tgt_gram - rec_gram)**2)
                norm = tf.reduce_sum(tgt_gram**2)

                loss_acc += cost / norm

            self.loss = loss_acc / len(layers)

    def get_score_from_files(self, src_file, rec_file):

        src_mat = load_image(src_file)
        if src_mat.shape[0] != 1:
            src_mat = np.expand_dims(src_mat, axis=0)
        rec_mat = load_image(rec_file, resize=False)
        if rec_mat.shape[0] != 1:
            rec_mat = np.expand_dims(rec_mat, axis=0)

        self.get_score(src_mat, rec_mat)

    def get_score(self, src_mat, rec_mat):
        with tf.Graph().as_default():
            src_name, rec_name = self.in_tensor_names
            src = tf.constant(src_mat, dtype=tf.float32, name=src_name[:-len(':0')])
            rec = tf.constant(rec_mat, dtype=tf.float32, name=rec_name[:-len(':0')])
            print(src, rec)
            self.build()

            with tf.Session() as sess:
                loss = self.get_loss()
                score = sess.run(loss)

        return score

