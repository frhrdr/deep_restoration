import tensorflow as tf
import os


class LossModule:

    def __init__(self, tensor_names, weighting):
        self.tensor_names = tensor_names
        self.weighting = weighting
        self.loss = None
        self.name = self.__class__.__name__

    def build(self):
        self.loss = 0

    def get_loss(self, weighted=True):
        if self.loss is not None:
            if weighted:
                return self.loss * self.weighting
            else:
                return self.loss
        else:
            raise AttributeError

    def get_tensors(self):
        g = tf.get_default_graph()
        if isinstance(self.tensor_names, tuple):
            return [g.get_tensor_by_name(n) for n in self.tensor_names]
        else:
            return g.get_tensor_by_name(self.tensor_names)

    def scalar_summary(self, weighted=True):
        tf.summary.scalar(self.name, self.get_loss(weighted))


class MSELoss(LossModule):

    def __init__(self, target, reconstruction, weighting):
        super().__init__((target, reconstruction), weighting)

    def build(self):
        tgt, rec = self.get_tensors()
        self.loss = tf.losses.mean_squared_error(tgt, rec)


class NormedMSELoss(MSELoss):

    def build(self):
        tgt, rec = self.get_tensors()
        self.loss = tf.reduce_sum((tgt - rec) ** 2) / tf.reduce_sum(tgt * rec)


class SoftRangeLoss(LossModule):

    def __init__(self, tensor, alpha, weighting):
        super().__init__(tensor, weighting)
        self.alpha = alpha

    def build(self):
        tensor = self.get_tensors()
        tmp = tf.reduce_sum(tensor ** 2, axis=[3]) ** (self.alpha / 2)
        self.loss = tf.reduce_sum(tmp)


class TotalVariationLoss(LossModule):

    def __init__(self, tensor, beta, weighting):
        super().__init__(tensor, weighting)
        self.beta = beta

    def build(self):
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


class LearnedPriorLoss(LossModule):

    def __init__(self, tensor_names, weighting, name, load_path, trainable):
        super().__init__(tensor_names, weighting)
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
