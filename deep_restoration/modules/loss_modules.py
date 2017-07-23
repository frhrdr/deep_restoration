import tensorflow as tf
import os


class Module:

    def __init__(self, in_tensor_names):
        self.in_tensor_names = in_tensor_names
        self.name = self.__class__.__name__

    def build(self, scope_suffix=''):
        raise NotImplementedError

    def get_in_tensors(self):
        g = tf.get_default_graph()
        if isinstance(self.in_tensor_names, tuple):
            return [g.get_tensor_by_name(n) for n in self.in_tensor_names]
        else:
            return g.get_tensor_by_name(self.in_tensor_names)


class LossModule(Module):

    def __init__(self, in_tensor_names, weighting):
        super().__init__(in_tensor_names)
        self.weighting = weighting
        self.loss = None

    def build(self, scope_suffix=''):
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
        if isinstance(self.in_tensor_names, tuple):
            return [g.get_tensor_by_name(n) for n in self.in_tensor_names]
        else:
            return g.get_tensor_by_name(self.in_tensor_names)

    def scalar_summary(self, weighted=True):
        tf.summary.scalar(self.name, self.get_loss(weighted))


class MSELoss(LossModule):

    def __init__(self, target, reconstruction, weighting):
        super().__init__((target, reconstruction), weighting)

    def build(self, scope_suffix=''):
        tgt, rec = self.get_tensors()
        self.loss = tf.losses.mean_squared_error(tgt, rec)


class NormedMSELoss(MSELoss):

    def build(self, scope_suffix=''):
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


class LearnedPriorLoss(LossModule):

    def __init__(self, tensor_names, weighting, name, load_path, trainable, load_name):
        super().__init__(tensor_names, weighting)
        self.name = name
        self.load_path = load_path
        self.trainable = trainable
        self.var_list = []
        self.load_name = load_name

    def load_weights(self, session):
        if self.load_name != '':

            names = [k.name.split('/')[1:] for k in self.var_list]
            names = [''.join(k) for k in names]
            names = [self.load_name + '/' + k.split(':')[0] for k in names]
            to_load = dict(zip(names, self.var_list))
        else:
            to_load = self.var_list
        loader = tf.train.Saver(var_list=to_load)
        loader.restore(session, self.load_path)

    def save_weights(self, session, step):
        saver = tf.train.Saver(var_list=self.var_list)
        checkpoint_file = os.path.join(self.load_path, 'ckpt')
        saver.save(session, checkpoint_file, global_step=step, write_meta_graph=False)
