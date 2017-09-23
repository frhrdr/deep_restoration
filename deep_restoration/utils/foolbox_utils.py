import tensorflow as tf
import numpy as np
from tf_alexnet.alexnet import AlexNet
from tf_vgg.vgg16 import Vgg16
import matplotlib
matplotlib.use('tkagg', force=True)
import matplotlib.pyplot as plt
from utils.temp_utils import load_image
from utils.imagenet_classnames import get_class_name
from modules.core_modules import LearnedPriorLoss
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
        input_pl = tf.placeholder(tf.float32, (1, 224, 224, 3))
        classifier = Vgg16()
        classifier.build(input_pl)
        logit_tsr = tf.get_default_graph().get_tensor_by_name('fc8/lin:0')
    else:
        input_pl = tf.placeholder(tf.float32, (1, 227, 227, 3))
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


def make_untargeted_examples(source_images, save_dir,
                             classifier='alexnet', attack_name='deepfool', attack_keys=None, verbose=False):
    for src_image in source_images:
        with tf.Graph().as_default():
            with tf.Session():
                input_pl, logit_tsr = get_classifier_io(classifier)
                model = foolbox.models.TensorFlowModel(input_pl, logit_tsr, bounds=(0, 255))

                if src_image.endswith('bmp') or src_image.endswith('png'):
                    image = load_image(src_image, resize=False)
                elif src_image.endswith('npy'):
                    image = np.load(src_image)
                else:
                    raise NotImplementedError

                pred = model.predictions(image)
                src_label = np.argmax(pred)
                if verbose:
                    src_label_name = get_class_name(src_label)
                    print('source image classified as {}. (label {})'.format(src_label_name, src_label))

                criterion = foolbox.criteria.Misclassification()
                attack = get_attack(attack_name, model, criterion)
                if attack_keys is None:
                    attack_keys = dict()
                adversarial = attack(image=image, label=src_label, **attack_keys)
                if adversarial is None:
                    print('no adversary found for source label {} using {}'.format(src_label, attack_name))
                    continue
                fooled_pred = model.predictions(adversarial)
                fooled_label = np.argmax(fooled_pred)
                if verbose:
                    fooled_label_name = get_class_name(fooled_label)
                    print('adversarial image classified as {}. (label {})'.format(fooled_label_name, fooled_label))
                save_file = get_adv_ex_filename(src_image, src_label, fooled_label, save_dir, file_type='bmp')
                if adversarial.dtype == np.float32:
                    adversarial = np.minimum(adversarial / 255, 1.0)

                skimage.io.imsave(save_file, adversarial)


def get_prior_scores_per_image(image_paths, priors, classifier='alexnet', verbose=True):

    with tf.Graph().as_default():

        img_pl, _ = get_classifier_io(classifier)
        loss_list = []
        loss_tsr = 0
        for prior in priors:
            prior.build()
            loss_tsr += prior.get_loss()

        with tf.Session() as sess:
            for prior in priors:
                if isinstance(prior, LearnedPriorLoss):
                    prior.load_weights(sess)

            for path in image_paths:
                image = np.expand_dims(load_image(path), axis=0)
                img_loss = sess.run(loss_tsr, feed_dict={img_pl: image})
                loss_list.append(img_loss)

                if verbose:
                    img_name = path.split('/')[-1].split('.')[0]
                    print('Loss {0:10.4f} for image {1}'.format(img_loss, img_name))
    return loss_list


def make_small_selected_dataset():
    image_indices = (53, 76, 81, 99, 106, 108, 129, 153, 157, 160)
    image_path = '../data/selected/images_resized_227/val{}.bmp'
    target_classes = [844, 424, 970, 949, 906, 486, 934, 39, 277, 646]
    attack_name = 'itgradientsign'
    save_dir = '../data/adversarial_examples/foolbox_images/small_dataset/{}/'.format(attack_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for idx in image_indices:
        make_targeted_examples(image_path.format(idx), target_classes, save_dir,
                               attack_name=attack_name, attack_keys={'epsilons': 1000, 'steps': 50}, verbose=True)


def make_small_untargeted_dataset():
    data_dir = '../data/imagenet2012-validationset/'
    images_file = 'subset_cutoff_200_images.txt'
    subdir = 'images_resized_227/'

    with open(data_dir + images_file) as f:
        image_files = [k.rstrip() for k in f.readlines()]
        image_paths = [data_dir + subdir + k[:-len('JPEG')] + 'bmp' for k in image_files]

    attack_name = 'deepfool'
    save_dir = '../data/adversarial_examples/foolbox_images/200_dataset/{}/'.format(attack_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    make_untargeted_examples(image_paths, save_dir, classifier='alexnet', attack_name='deepfool',
                             attack_keys=None, verbose=True)


def compare_images_to_untargeted_adv_ex(priors):
    data_dir = '../data/imagenet2012-validationset/'
    images_file = 'subset_cutoff_200_images.txt'
    subdir = 'images_resized_227/'
    with open(data_dir + images_file) as f:
        image_files = [k.rstrip() for k in f.readlines()]
        image_paths = [data_dir + subdir + k[:-len('JPEG')] + 'bmp' for k in image_files]

    image_losses = get_prior_scores_per_image(image_paths, priors, classifier='alexnet', verbose=False)

    advex_dir = '../data/adversarial_examples/foolbox_images/200_dataset/deepfool/'
    image_stems = [k[:-len('.JPEG')] for k in image_files]
    advex_files = sorted(os.listdir(advex_dir))
    advex_paths = [advex_dir + k for k in advex_files]
    advex_losses = get_prior_scores_per_image(advex_paths, priors, classifier='alexnet', verbose=False)
    img_idx = 0
    advex_matches = []
    for adv_idx, adv_file in enumerate(advex_files):
        while not adv_file.startswith(image_stems[img_idx]):
            img_idx += 1
        advex_matches.append((image_losses[img_idx], advex_losses[adv_idx]))
    print('number of matches:', len(advex_matches))
    diff = np.asarray([k[0] - k[1] for k in advex_matches])
    print(diff)

    print('mean difference:', np.mean(diff))
    print('number of images with higher ad-ex loss', np.sum(diff < 0))
    print('number of images with lower or equal ad-ex loss', np.sum(diff >= 0))
