import tensorflow as tf
import numpy as np
from tf_alexnet.alexnet import AlexNet
from tf_vgg.vgg16 import Vgg16
import matplotlib
matplotlib.use('tkagg', force=True)
import matplotlib.pyplot as plt
from utils.temp_utils import load_image, get_optimizer
from utils.imagenet_classnames import get_class_name
from modules.core_modules import LearnedPriorLoss
from modules.foe_full_prior import FoEFullPrior
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


def get_classifier_io(name, input_init=None, input_type='placeholder'):
    assert name in ('alexnet', 'vgg16')
    assert input_type in ('placeholder', 'variable', 'constant', 'tensor')
    if name == 'vgg16':
        classifier = Vgg16()
        hw = 224
    else:
        classifier = AlexNet()
        hw = 227

    if input_type == 'placeholder':
        input_tensor = tf.placeholder(tf.float32, (1, hw, hw, 3))
    elif input_type == 'variable':
        input_tensor = tf.get_variable('input_image', initializer=input_init, dtype=tf.float32)
    elif input_type == 'constant':
        input_tensor = tf.constant(input_init, dtype=tf.float32)
    else:
        input_tensor = input_init

    classifier.build(input_tensor)
    logit_tsr = tf.get_default_graph().get_tensor_by_name('fc8/lin:0')

    return input_tensor, logit_tsr


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
                image = image.astype(np.float32)

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
                save_file = get_adv_ex_filename(src_image, src_label, fooled_label, save_dir, file_type='npy')
                # if adversarial.dtype == np.float32:
                #     adversarial = np.minimum(adversarial / 255, 1.0)

                np.save(save_file, adversarial)
                # skimage.io.imsave(save_file, adversarial)


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
                             attack_keys={'steps': 300}, verbose=True)


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
    print('number of images with equal ad-ex loss', np.sum(diff == 0))
    print('number of images with lower ad-ex loss', np.sum(diff > 0))

    print(np.max(advex_losses))
    print(np.min(advex_losses))
    print(np.max(image_losses))
    print(np.min(image_losses))


def eval_class_stability(image_file, priors, learning_rate, n_iterations, log_freq,
                         optimizer='sgd', classifier='alexnet', verbose=False):
    if not isinstance(log_freq, list):
        log_freq = list(range(log_freq, n_iterations, log_freq))
    log_list = []

    if image_file.endswith('bmp') or image_file.endswith('png'):
        image = load_image(image_file, resize=False)
    elif image_file.endswith('npy'):
        image = np.load(image_file)
    else:
        raise NotImplementedError
    image = np.expand_dims(image.astype(dtype=np.float32), axis=0)
    # print(image)
    with tf.Graph().as_default():

        input_var, logit_tsr = get_classifier_io(classifier, input_init=image, input_type='variable')

        loss_tsr = 0
        for prior in priors:
            prior.build()
            loss_tsr += prior.get_loss()

        optimizer = get_optimizer(optimizer, learning_rate)
        train_op = optimizer.minimize(loss_tsr)

        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            for prior in priors:
                if isinstance(prior, LearnedPriorLoss):
                    prior.load_weights(sess)

            pred = sess.run(logit_tsr)
            current_label = np.argmax(pred)
            log_list.append(current_label)
            if verbose:
                current_label_name = get_class_name(current_label)
                print('image initially classified as {}. (label {})'.format(current_label_name, current_label))

            for count in range(n_iterations + 1):
                _, loss = sess.run([train_op, loss_tsr])

                if log_freq and count == log_freq[0]:
                    log_freq = log_freq[1:]
                    pred = sess.run(logit_tsr)
                    new_label = np.argmax(pred)
                    log_list.append(new_label)
                    if verbose and new_label != current_label:
                        new_label_name = get_class_name(new_label)
                        print(('label changed at iteration {}: ' +
                              'now classified as {} (label {})').format(count, new_label_name, new_label))
                        print('Loss:', loss)
                    current_label = new_label
    return log_list


def advex_match_paths_200():
    data_dir = '../data/imagenet2012-validationset/'
    images_file = 'subset_cutoff_200_images.txt'
    subdir = 'images_resized_227/'
    advex_dir = '../data/adversarial_examples/foolbox_images/200_dataset/deepfool_oblivious/'

    with open(data_dir + images_file) as f:
        image_files = [k.rstrip() for k in f.readlines()]
        image_paths = [data_dir + subdir + k[:-len('JPEG')] + 'bmp' for k in image_files]

    image_stems = [k[:-len('.JPEG')] for k in image_files]
    advex_files = sorted(os.listdir(advex_dir))
    advex_paths = [advex_dir + k for k in advex_files]
    img_idx = 0
    advex_matches = []
    for adv_idx, adv_file in enumerate(advex_files):
        while not adv_file.startswith(image_stems[img_idx]):
            img_idx += 1
        advex_matches.append((image_paths[img_idx], advex_paths[adv_idx]))
    return advex_matches


def stability_experiment_200():
    imgprior = FoEFullPrior('rgb_scaled:0', 1e-5, 'alexnet', [8, 8], 1.0, n_components=512, n_channels=3,
                            n_features_white=8 ** 2 * 3 - 1, dist='student', mean_mode='gc', sdev_mode='gc',
                            whiten_mode='pca',
                            name=None, load_name='FoEPrior', dir_name=None, load_tensor_names='image')
    optimizer = 'adam'
    # learning_rate = 1e-0
    # n_iterations = 100
    # log_freq = list(range(1, 5)) + list(range(5, 50, 5)) + list(range(50, 101, 10))
    # learning_rate = 1e-1
    # n_iterations = 20
    # log_freq = 1

    learning_rate = 1e-0
    n_iterations = 1000
    log_freq = 100

    advex_matches = advex_match_paths_200()
    print('number of matches:', len(advex_matches))
    count = 0
    img_list = []
    adv_list = []
    for img_path, adv_path in advex_matches[:100]:
        count += 1
        print('match no.', count)
        log_list = eval_class_stability(img_path, [imgprior], learning_rate, n_iterations, log_freq,
                                        optimizer=optimizer, classifier='alexnet', verbose=True)
        img_list.append(log_list)
        imgprior.reset()
        log_list = eval_class_stability(adv_path, [imgprior], learning_rate, n_iterations, log_freq,
                                        optimizer=optimizer, classifier='alexnet', verbose=True)
        adv_list.append(log_list)
        imgprior.reset()
    print(img_list)
    print(adv_list)
    np.save('img_log_sgd_1.npy', np.asarray(img_list))
    np.save('adv_log_sgd_1.npy', np.asarray(adv_list))


def stability_statistics():
    log_freq = list(range(1, 5)) + list(range(5, 50, 5)) + list(range(50, 101, 10))
    print('log points after n iterations', log_freq)

    path = '../logs/adversarial_examples/deepfool_oblivious_198/'
    img_log = np.load(path + 'img_log_198_fine.npy')
    adv_log = np.load(path + 'adv_log_198_fine.npy')

    print(img_log.shape)
    n_samples, n_logpoints = img_log.shape
    # count changes over channels by log point
    img_changes = np.minimum(np.abs(img_log[:, :-1] - img_log[:, 1:]), 1)
    adv_changes = np.minimum(np.abs(adv_log[:, :-1] - adv_log[:, 1:]), 1)

    img_changes_sum = np.sum(img_changes, axis=0)
    adv_changes_sum = np.sum(adv_changes, axis=0)
    print('count of changes in images', list(img_changes_sum))
    print('count of changes in adv ex', list(adv_changes_sum))

    # count first change only

    img_count = np.argmax(img_changes, axis=1)
    adv_count = np.argmax(adv_changes, axis=1)

    img_range = np.zeros(n_logpoints - 1)
    adv_range = np.zeros(n_logpoints - 1)

    for idx in range(n_samples):
        img_range[img_count[idx]] += img_changes[idx, img_count[idx]]
        adv_range[adv_count[idx]] += adv_changes[idx, adv_count[idx]]

    print('count of first changes in images', list(img_range.astype(np.int)))
    print('count of first changes in adv ex', list(adv_range.astype(np.int)))

    # find reversion to original label
    src_labels = img_log[:, 0]
    src_found = 1 - np.minimum(np.abs(adv_log.T - src_labels).T, 1)
    src_count = np.argmax(src_found, axis=1)

    src_range = np.zeros(n_logpoints)

    for idx in range(n_samples):
        assert src_labels[idx] == adv_log[idx, src_count[idx]] or src_found[idx, src_count[idx]] == 0
        src_range[src_count[idx]] += src_found[idx, src_count[idx]]

    print('count of first changes to original label in adv ex', list(src_range.astype(np.int))[1:])

    # plt.plot(log_freq, src_range[1:], 'ro')
    # plt.show()


def eval_whitebox_forward_opt(image, prior, learning_rate, n_iterations, attack_name, attack_keys, src_label,
                              classifier='alexnet', verbose=False):
    """
    runs baseline with prior, then whitebox attack
    :param image:
    :param prior:
    :param learning_rate:
    :param n_iterations:
    :param attack_name:
    :param attack_keys:
    :param src_label: label given by classifier without prior
    :param classifier:
    :param verbose:
    :return:
    """

    with tf.Graph().as_default():
        # input_featmap = tf.constant(image, dtype=tf.float32)
        input_featmap = tf.placeholder(tf.float32, image.shape)
        featmap = prior.forward_opt_sgd(input_featmap, learning_rate, n_iterations)
        _, logit_tsr = get_classifier_io(classifier, input_init=featmap, input_type='tensor')

        with tf.Session() as sess:
            model = foolbox.models.TensorFlowModel(input_featmap, logit_tsr, bounds=(0, 255))
            criterion = foolbox.criteria.Misclassification()
            attack = get_attack(attack_name, model, criterion)
            if attack_keys is None:
                attack_keys = dict()

            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            prior.load_weights(sess)
            pred = model.predictions(image)
            noisy_label = np.argmax(pred)

            if verbose:
                noisy_label_name = get_class_name(noisy_label)
                if noisy_label == src_label:
                    print('noisy label {} same as source: {}'.format(noisy_label, noisy_label_name))
                else:
                    print('image with prior misclassified as {}. (label {})'.format(noisy_label_name, noisy_label))
            adversarial = attack(image=image, label=noisy_label, **attack_keys)
            if adversarial is None:
                print('no adversary found for source label {} using {}'.format(noisy_label, attack_name))
                return None
            else:
                fooled_pred = model.predictions(adversarial)
                fooled_label = np.argmax(fooled_pred)
                fooled_label_name = get_class_name(fooled_label)
                noise_norm = np.linalg.norm((image - adversarial).flatten(), ord=2)
                if verbose:
                    print('adversarial image classified as {}. (label {}) '
                          'Necessary perturbation: {}'.format(fooled_label_name, fooled_label, noise_norm))
                return adversarial, noise_norm


def whitebox_experiment_200(learning_rate=0.1, n_iterations=2, attack_name='deepfool', attack_keys=None, verbose=True):
    path = '../logs/adversarial_examples/deepfool_oblivious_198/'
    img_log = np.load(path + 'img_log_198_fine.npy')
    # adv_log = np.load(path + 'adv_log_198_fine.npy')
    classifier = 'alexnet'
    image_shape = (1, 227, 227, 3)
    advex_matches = advex_match_paths_200()

    imgprior = FoEFullPrior('rgb_scaled:0', 1e-5, 'alexnet', [8, 8], 1.0, n_components=512, n_channels=3,
                            n_features_white=8 ** 2 * 3 - 1, dist='student', mean_mode='gc', sdev_mode='gc',
                            whiten_mode='pca',
                            name=None, load_name='FoEPrior', dir_name=None, load_tensor_names='image')

    with tf.Graph().as_default():
        # input_featmap = tf.constant(image, dtype=tf.float32)
        input_featmap = tf.placeholder(dtype=tf.float32, shape=image_shape)
        featmap = imgprior.forward_opt_adam(input_featmap, learning_rate, n_iterations)
        _, logit_tsr = get_classifier_io(classifier, input_init=featmap, input_type='tensor')

        with tf.Session() as sess:
            model = foolbox.models.TensorFlowModel(input_featmap, logit_tsr, bounds=(0, 255))
            criterion = foolbox.criteria.Misclassification()
            attack = get_attack(attack_name, model, criterion)
            if attack_keys is None:
                attack_keys = dict()

            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            imgprior.load_weights(sess)

            noise_norms = []
            src_invariant = []
            for idx, match in enumerate(advex_matches[:2]):
                img_path, adv_path = match
                src_label = img_log[idx][0]

                img = load_image(img_path)
                adv = load_image(adv_path)

                oblivious_norm = np.linalg.norm((img - adv).flatten(), ord=2)
                print('oblivious norm', oblivious_norm)

                pred = model.predictions(img)
                noisy_label = np.argmax(pred)

                if verbose:
                    noisy_label_name = get_class_name(noisy_label)
                    if noisy_label == src_label:
                        print('noisy label {} same as source: {}'.format(noisy_label, noisy_label_name))
                        src_invariant.append(1)
                    else:
                        print('image with prior misclassified as {}. (label {})'.format(noisy_label_name, noisy_label))
                        src_invariant.append(0)
                adversarial = attack(image=img, label=noisy_label, **attack_keys)
                if adversarial is None:
                    whitebox_norm = None
                    if verbose:
                        print('no adversary found for source label {} using {}'.format(noisy_label, attack_name))
                else:
                    fooled_pred = model.predictions(adversarial)
                    fooled_label = np.argmax(fooled_pred)
                    fooled_label_name = get_class_name(fooled_label)
                    whitebox_norm = np.linalg.norm((img - adversarial).flatten(), ord=2)
                    if verbose:
                        print('adversarial image classified as {}. (label {}) '
                              'Necessary perturbation: {}'.format(fooled_label_name, fooled_label, whitebox_norm))
                    whitebox_save_path = adv_path.replace('oblivious', 'whitebox')
                    np.save(whitebox_save_path, adversarial)
                noise_norms.append((oblivious_norm, whitebox_norm))
