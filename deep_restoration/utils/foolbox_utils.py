import tensorflow as tf
import numpy as np
from tf_alexnet.alexnet import AlexNet
from tf_vgg.vgg16 import Vgg16
import matplotlib
matplotlib.use('tkagg', force=True)
import matplotlib.pyplot as plt
from utils.temp_utils import load_image
from utils.imagenet_classnames import get_class_name
import foolbox
import skimage.io
import os


def get_attack(name, model, criterion):
    attacks = {'lbfgs': foolbox.attacks.LBFGSAttack,
               'gradient': foolbox.attacks.GradientAttack,
               'itgradient': foolbox.attacks.IterativeGradientAttack,
               'gradientsign': foolbox.attacks.GradientSignAttack,
               'itgradientsign': foolbox.attacks.IterativeGradientSignAttack,
               'deepfool': foolbox.attacks.DeepFoolAttack,
               'saliencymap': foolbox.attacks.SaliencyMapAttack}
    assert name in attacks
    return attacks[name](model, criterion)


def get_classifier_io(name):
    assert name in ('alexnet', 'vgg16')
    if name == 'vgg16':
        input_pl = tf.placeholder(tf.float32, (None, 224, 224, 3))
        classifier = Vgg16()
        classifier.build(input_pl)
        logit_tsr = tf.get_default_graph().get_tensor_by_name('fc8/lin:0')
    else:
        input_pl = tf.placeholder(tf.float32, (None, 227, 227, 3))
        classifier = AlexNet()
        classifier.build(input_pl)
        logit_tsr = tf.get_default_graph().get_tensor_by_name('fc8/lin:0')
    return input_pl, logit_tsr


def get_adv_ex_filename(source_image, source_label, target_label, save_dir, file_type='png'):
    file_name = source_image.split('/')[-1].split('.')[0]
    return '{}/{}_t{}_f{}.{}'.format(save_dir, file_name, source_label, target_label, file_type)


def make_targeted_examples(source_image, target_class_labels, save_dir,
                           classifier='alexnet', attack_name='lbfgs', attack_keys=None, verbose=False):
    src_label = None
    for tgt_label in target_class_labels:
        with tf.Graph().as_default():
            with tf.Session():
                input_pl, logit_tsr = get_classifier_io(classifier)
                model = foolbox.models.TensorFlowModel(input_pl, logit_tsr, bounds=(0, 255))

                if source_image.endswith('bmp') or source_image.endswith('png'):
                    image = load_image(source_image, resize=False)
                elif source_image.endswith('npy'):
                    image = np.load(source_image)
                else:
                    raise NotImplementedError

                if src_label is None:
                    pred = model.predictions(image)
                    src_label = np.argmax(pred)
                    if verbose:
                        src_label_name = get_class_name(src_label)
                        print('source image classified as {}. (label {})'.format(src_label_name, src_label))

                criterion = foolbox.criteria.TargetClass(tgt_label)
                attack = get_attack(attack_name, model, criterion)
                if attack_keys is None:
                    attack_keys = dict()
                adversarial = attack(image=image, label=tgt_label, **attack_keys)
                if adversarial is None:
                    print('no adversary found for source label {} and target {} using {}'.format(src_label, tgt_label,
                                                                                                 attack_name))
                    continue
                if verbose:
                    print(adversarial)
                    fooled_pred = model.predictions(adversarial)
                    fooled_label = np.argmax(fooled_pred)
                    fooled_label_name = get_class_name(fooled_label)
                    print('adversarial image classified as {}. (label {})'.format(fooled_label_name, fooled_label))

                save_file = get_adv_ex_filename(source_image, src_label, tgt_label, save_dir, file_type='bmp')
                skimage.io.imsave(save_file, adversarial)


def make_small_selected_dataset():
    image_indices = (53, 76, 81, 99, 106, 108, 129, 153, 157, 160)
    image_path = '../data/selected/images_resized_227/val{}.bmp'
    target_classes = [844, 424, 970, 949, 906, 486, 934, 39, 277, 646]
    attack_name = 'lbfgs'
    save_dir = '../data/adversarial_examples/foolbox_images/small_dataset/{}/'.format(attack_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for idx in image_indices:
        make_targeted_examples(image_path.format(idx), target_classes, save_dir,
                               attack_name=attack_name, attack_keys={'maxiter': 1000}, verbose=True)

