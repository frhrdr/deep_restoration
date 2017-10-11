import tensorflow as tf
from tf_alexnet.alexnet import AlexNet
from cleverhans.attacks import FastGradientMethod
from cleverhans.model import Model


class AlexnetCHModel(Model):

    def __init__(self):
        super().__init__()
        self.layer_names = ('logits', 'probs')
        self.net = AlexNet(make_dict=True)

    def fprop(self, x):

        self.net.build(x)
        return {'logits': self.net.tensors['fc8/lin'], 'probs': self.net.tensors['softmax']}


sess = tf.Session()

img_pl = tf.placeholder(dtype=tf.float32, shape=(1, 227, 227, 3))

anet = AlexnetCHModel()
fgsm = FastGradientMethod(anet, sess=sess)

fgsm_params = {'eps': 0.3,
               'clip_min': 0.,
               'clip_max': 255.}

x_pl = tf.placeholder(tf.float32, shape=(None, 227, 227, 3))

adv_x = fgsm.generate(x_pl, **fgsm_params)
preds_adv = anet.get_probs(adv_x)


