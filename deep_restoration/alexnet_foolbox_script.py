import numpy as np
import tensorflow as tf
from tf_alexnet.alexnet import AlexNet
from tf_vgg.vgg16 import Vgg16
import matplotlib
matplotlib.use('tkagg', force=True)
import matplotlib.pyplot as plt
from utils.temp_utils import load_image
from foolbox.models import TensorFlowModel
from foolbox.criteria import TargetClass, Misclassification
from foolbox.attacks import LBFGSAttack, GaussianBlurAttack

with tf.Graph().as_default() as graph:
    with tf.Session() as sess:
        images = tf.placeholder(tf.float32, (None, 227, 227, 3))
        net = AlexNet()
        net.build(images, rescale=1.0)
        # logits = graph.get_tensor_by_name('softmax:0')
        logits = graph.get_tensor_by_name('fc8/relu:0')

        model = TensorFlowModel(images, logits, bounds=(0, 255))

        target_class = 280
        criterion = TargetClass(target_class)
        # criterion = Misclassification()
        attack = LBFGSAttack(model, criterion)
        # attack = GaussianBlurAttack(model, criterion)

        image_path = '../data/selected/images_resized_227/red-fox.bmp'
        image = load_image(image_path, resize=False)
        pred = model.predictions(image)
        label = np.argmax(pred)
        confidence = np.max(pred)
        print(label, confidence)
        adversarial = attack(image=image, label=label)
        fooled_pred = model.predictions(adversarial)
        fooled_label = np.argmax(fooled_pred)
        fooled_confidence = np.max(fooled_pred)
        print(fooled_label, fooled_confidence)


plt.subplot(1, 3, 1)
plt.imshow(image)

plt.subplot(1, 3, 2)
plt.imshow(adversarial)

plt.subplot(1, 3, 3)
plt.imshow(adversarial - image)

plt.show()
