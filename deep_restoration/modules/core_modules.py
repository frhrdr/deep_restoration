import os
import tensorflow as tf


class Module:

    def __init__(self, in_tensor_names, name=None):
        self.in_tensor_names = in_tensor_names
        self.name = name if name is not None else self.__class__.__name__

    def build(self, scope_suffix=''):
        raise NotImplementedError

    def get_tensors(self):
        g = tf.get_default_graph()
        if isinstance(self.in_tensor_names, tuple):
            return tuple(g.get_tensor_by_name(n) for n in self.in_tensor_names)
        else:
            return g.get_tensor_by_name(self.in_tensor_names)


class LossModule(Module):

    def __init__(self, in_tensor_names, weighting, name=None):
        super().__init__(in_tensor_names, name=name)
        self.weighting = weighting
        self.loss = None
        self.weighted_loss = None
        self.add_loss = True

    def reset(self):
        self.loss = None
        self.weighted_loss = None

    def build(self, scope_suffix=''):
        self.loss = 0

    def get_loss(self, weighted=True):
        assert self.loss is not None
        if self.add_loss:
            return self.get_weighted_loss() if weighted else self.loss
        else:
            return 0

    def get_weighted_loss(self):
        if self.weighted_loss is None:
            self.weighted_loss = self.loss * self.weighting
        return self.weighted_loss

    def scalar_summary(self, weighted=True):
        loss = self.get_weighted_loss() if weighted else self.loss
        tf.summary.scalar(self.name, loss)


class LearnedPriorLoss(LossModule):

    def __init__(self, tensor_names, weighting, name, load_path, load_name):
        super().__init__(tensor_names, weighting)
        self.name = name
        self.load_path = load_path
        self.var_list = []
        self.load_name = load_name

    def load_weights(self, session, ckpt_id=None):
        to_load = self.tensor_load_dict_by_name(self.var_list)
        loader = tf.train.Saver(var_list=to_load)

        if ckpt_id is None:
            with open(os.path.join(self.load_path, 'checkpoint')) as f:
                ckpt = f.readline().split('"')[1]
        else:
            ckpt = 'ckpt-' + str(ckpt_id)
        print('For module {0}: loading weights from {1}'.format(self.name, ckpt))
        loader.restore(session, os.path.join(self.load_path, ckpt))

    def save_weights(self, session, step, checkpoint_name='ckpt'):
        saver = tf.train.Saver(var_list=self.var_list)
        checkpoint_file = os.path.join(self.load_path, checkpoint_name)
        saver.save(session, checkpoint_file, global_step=step, write_meta_graph=False)

    def tensor_load_dict_by_name(self, tensor_list):
        names = [k.name.split('/') for k in tensor_list]
        prev_load_name = names[0][0]
        names = [[l if l != prev_load_name else self.load_name for l in k] for k in names]
        names = ['/'.join(k) for k in names]
        names = [k.split(':')[0] for k in names]
        return dict(zip(names, tensor_list))


class TrainedModule(Module):

    def __init__(self, tensor_names, name, load_path, load_name, trainable):
        super().__init__(tensor_names)
        self.name = name
        self.load_path = load_path
        self.load_name = load_name
        self.trainable = trainable
        self.var_list = []

    def save_weights(self, session, step, checkpoint_name='ckpt'):
        if not os.path.exists(self.load_path):
            os.makedirs(self.load_path)

        to_save = self.tensor_load_dict_by_name(self.var_list)
        saver = tf.train.Saver(var_list=to_save)
        checkpoint_file = os.path.join(self.load_path, checkpoint_name)
        saver.save(session, checkpoint_file, global_step=step, write_meta_graph=False)

    def load_weights(self, session):
        to_load = self.tensor_load_dict_by_name(self.var_list)
        loader = tf.train.Saver(var_list=to_load)
        print(self.load_path)
        with open(os.path.join(self.load_path, 'checkpoint')) as f:
            ckpt = f.readline().split('"')[1]
            print('For module {0}: loading weights from {1}'.format(self.name, ckpt))
        loader.restore(session, os.path.join(self.load_path, ckpt))

    def tensor_load_dict_by_name(self, tensor_list):
        names = [k.name.split('/') for k in tensor_list]
        prev_load_name = names[0][0]
        names = [[l if l != prev_load_name else self.load_name for l in k] for k in names]
        names = ['/'.join(k) for k in names]
        names = [k.split(':')[0] for k in names]
        return dict(zip(names, tensor_list))

    def build(self, scope_suffix=''):
        raise NotImplementedError

    def is_trainable(self):
        return self.trainable


class InversionModule(TrainedModule):

    def __init__(self, inv_input_name, inv_target_name,
                 hidden_channels, rec_name,
                 op1_hw, op1_strides, op2_hw, op2_strides, input_from_rec,
                 op1_pad='SAME', op2_pad='SAME', name='InversionModule',
                 dir_name=None, load_name=None, subdir='',
                 trainable=False, alt_load_subdir=None):

        dir_name = dir_name or name
        load_name = load_name or name
        load_path = self.get_load_path(dir_name, inv_input_name, inv_target_name, subdir)
        if input_from_rec is not None:
            in_tensors = (input_from_rec, inv_target_name)
        else:
            in_tensors = (inv_input_name, inv_target_name)

        super().__init__(in_tensors, name, load_path, load_name, trainable)

        self.hidden_channels = hidden_channels
        self.rec_name = rec_name
        self.op1_height = op1_hw[0]
        self.op1_width = op1_hw[1]
        self.op1_strides = op1_strides
        self.op1_pad = op1_pad
        self.op2_height = op2_hw[0]
        self.op2_width = op2_hw[1]
        self.op2_strides = op2_strides
        self.op2_pad = op2_pad
        self.subdir = subdir
        self.alt_load_subdir = alt_load_subdir

    def load_weights(self, session):
        save_path = self.load_path
        if self.alt_load_subdir is not None:
            self.load_path.replace(self.subdir, self.alt_load_subdir)
            print(self.load_path)
        super().load_weights(session=session)
        self.load_path = save_path

    def build(self, scope_suffix=''):
        raise NotImplementedError

    @staticmethod
    def get_load_path(name, inv_input_name, inv_target_name, subdir):
        io_string = inv_input_name.replace('/', '_').rstrip(':0') + '_to_' + \
                    inv_target_name.replace('/', '_').rstrip(':0')
        if subdir:
            subdir = subdir + '/'
        return '../logs/cnn_modules/{}/{}/{}'.format(name, io_string, subdir)
