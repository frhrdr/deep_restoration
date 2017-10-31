import os
import foolbox
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import skimage.io
import tensorflow as tf
from tf_alexnet.alexnet import AlexNet
from modules.core_modules import LearnedPriorLoss
from modules.foe_dropout_prior import FoEDropoutPrior
from utils.default_priors import get_default_prior
from utils.imagenet_classnames import get_class_name
from utils.temp_utils import load_image, get_optimizer


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
    # if name == 'vgg16':
    #     classifier = Vgg16()
    #     hw = 224
    # else:
    assert name == 'alexnet'
    classifier = AlexNet(make_dict=True)
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
    logit_tsr = classifier.tensors['fc8/lin']

    return input_tensor, logit_tsr, classifier


def get_adv_ex_filename(source_image, source_label, target_label, save_dir, file_type='png'):
    file_name = source_image.split('/')[-1].split('.')[0]
    return '{}/{}_t{}_f{}.{}'.format(save_dir, file_name, source_label, target_label, file_type)


def make_targeted_examples(source_image, target_class_labels, save_dir,
                           classifier='alexnet', attack_name='lbfgs', attack_keys=None, verbose=False):
    src_label = None
    for tgt_label in target_class_labels:
        with tf.Graph().as_default():
            with tf.Session():
                input_pl, logit_tsr, _ = get_classifier_io(classifier)
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
                input_pl, logit_tsr, _ = get_classifier_io(classifier)
                model = foolbox.models.TensorFlowModel(input_pl, logit_tsr, bounds=(0, 255))

                image = load_image(src_image)

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
                    # print('no adversary found for source label {} using {}'.format(src_label, attack_name))
                    print('no adversary found for source image {}'.format(src_image))
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

        img_pl, _, _ = get_classifier_io(classifier)
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
                             attack_keys={'steps': 300}, verbose=False)


def make_untargeted_dataset(image_subset, attack_name, attack_keys=None):
    if attack_name == 'deepfool':
        attack_keys = {'steps': 300}

    data_dir = '../data/imagenet2012-validationset/'
    images_subdir = 'images_resized_227/'

    with open(data_dir + image_subset) as f:
        image_paths = [k.rstrip() for k in f.readlines()]
        if '/' not in image_paths[0]:
            image_paths = [data_dir + images_subdir + k[:-len('JPEG')] + 'bmp' for k in image_paths]

    save_dir = '../data/adversarial_examples/foolbox_images/alexnet_val_2k_top1_correct' \
               '/{}_oblivious/'.format(attack_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    make_untargeted_examples(image_paths, save_dir, classifier='alexnet', attack_name=attack_name,
                             attack_keys=attack_keys, verbose=False)


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


def eval_class_stability(image_files, priors, learning_rate, n_iterations, log_freq,
                         optimizer='adam', classifier='alexnet', verbose=False):
    if not isinstance(log_freq, list):
        log_freq = list(range(log_freq, n_iterations, log_freq))
    log_list = []

    if classifier == 'alexnet':
        image_shape = (1, 227, 227, 3)
    else:
        image_shape = (1, 224, 224, 3)

    with tf.Graph().as_default():
        image_pl = tf.placeholder(dtype=tf.float32, shape=image_shape)
        image_var = tf.get_variable('input_image', shape=image_shape, dtype=tf.float32)
        _, logit_tsr, _ = get_classifier_io(classifier, input_init=image_var, input_type='tensor')
        feed_op = tf.assign(image_var, image_pl)

        loss_tsr = 0
        for prior in priors:
            prior.build()
            loss_tsr += prior.get_loss()

        optimizer = get_optimizer(optimizer, learning_rate)
        train_op = optimizer.minimize(loss_tsr)

        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            for img_no, file in enumerate(image_files):
                print('image no.', img_no)
                image_mat = load_image(file, resize=False)
                image_mat = np.expand_dims(image_mat.astype(dtype=np.float32), axis=0)
                image_log_list = []

                sess.run(init_op)
                for prior in priors:
                    if isinstance(prior, LearnedPriorLoss):
                        prior.load_weights(sess)
                sess.run(feed_op, feed_dict={image_pl: image_mat})

                pred = sess.run(logit_tsr)
                current_label = np.argmax(pred)
                image_log_list.append(current_label)
                if verbose:
                    current_label_name = get_class_name(current_label)
                    print('image initially classified as {}. (label {})'.format(current_label_name, current_label))

                log_idx = 0
                for count in range(n_iterations + 1):
                    _, loss = sess.run([train_op, loss_tsr])

                    if log_idx < len(log_freq) and count == log_freq[log_idx]:
                        log_idx += 1
                        pred = sess.run(logit_tsr)
                        new_label = np.argmax(pred)
                        image_log_list.append(new_label)
                        if verbose and new_label != current_label:
                            new_label_name = get_class_name(new_label)
                            print(('label changed at iteration {}: ' +
                                  'now classified as {} (label {})').format(count, new_label_name, new_label))
                            print('Loss:', loss)
                        current_label = new_label
                log_list.append(image_log_list)
    return log_list


def eval_class_stability_rolled_out(image_files, priors, learning_rate, n_iterations, log_freq,
                                    optimizer='adam', classifier='alexnet', verbose=False):
    if not isinstance(log_freq, list):
        log_freq = list(range(log_freq, n_iterations, log_freq))
    log_list = []

    if classifier == 'alexnet':
        image_shape = (1, 227, 227, 3)
    else:
        image_shape = (1, 224, 224, 3)

    with tf.Graph().as_default():
        image_pl = tf.placeholder(dtype=tf.float32, shape=image_shape)
        image_var = tf.get_variable('input_image', shape=image_shape, dtype=tf.float32)
        _, logit_tsr, _ = get_classifier_io(classifier, input_init=image_var, input_type='tensor')
        feed_op = tf.assign(image_var, image_pl)

        loss_tsr = 0
        for prior in priors:
            prior.build()
            loss_tsr += prior.get_loss()

        optimizer = get_optimizer(optimizer, learning_rate)
        train_op = optimizer.minimize(loss_tsr)

        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            for img_no, file in enumerate(image_files):
                print('image no.', img_no)
                image_mat = load_image(file, resize=False)
                image_mat = np.expand_dims(image_mat.astype(dtype=np.float32), axis=0)
                image_log_list = []

                sess.run(init_op)
                for prior in priors:
                    if isinstance(prior, LearnedPriorLoss):
                        prior.load_weights(sess)
                sess.run(feed_op, feed_dict={image_pl: image_mat})

                pred = sess.run(logit_tsr)
                current_label = np.argmax(pred)
                image_log_list.append(current_label)
                if verbose:
                    current_label_name = get_class_name(current_label)
                    print('image initially classified as {}. (label {})'.format(current_label_name, current_label))

                log_idx = 0
                for count in range(n_iterations + 1):
                    _, loss = sess.run([train_op, loss_tsr])

                    if log_idx < len(log_freq) and count == log_freq[log_idx]:
                        log_idx += 1
                        pred = sess.run(logit_tsr)
                        new_label = np.argmax(pred)
                        image_log_list.append(new_label)
                        if verbose and new_label != current_label:
                            new_label_name = get_class_name(new_label)
                            print(('label changed at iteration {}: ' +
                                  'now classified as {} (label {})').format(count, new_label_name, new_label))
                            print('Loss:', loss)
                        current_label = new_label
                log_list.append(image_log_list)
    return log_list


def advex_match_paths(images_file, advex_subdir):
    data_dir = '../data/imagenet2012-validationset/'
    image_subdir = 'images_resized_227/'
    advex_dir = '../data/adversarial_examples/foolbox_images/' + advex_subdir

    with open(data_dir + images_file) as f:
        image_files = [k.rstrip() for k in f.readlines()]

    if '/' not in image_files[0]:
        image_paths = [data_dir + image_subdir + k[:-len('JPEG')] + 'bmp' for k in image_files]
    else:
        image_paths = image_files

    image_stems = [k.split('/')[-1][:-len('.bmp')] for k in image_paths]
    advex_files = sorted(os.listdir(advex_dir))
    advex_paths = [advex_dir + k for k in advex_files]
    img_idx = 0
    advex_matches = []
    for adv_idx, adv_file in enumerate(advex_files):
        while not adv_file.startswith(image_stems[img_idx]):
            img_idx += 1
        advex_matches.append((image_paths[img_idx], advex_paths[adv_idx]))
    return advex_matches


def stability_experiment(images_file, advex_subdir, imgprior, optimizer, learning_rate,
                         n_iterations, log_freq, log_path):

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    advex_matches = advex_match_paths(images_file=images_file, advex_subdir=advex_subdir)
    print('number of matches:', len(advex_matches))

    img_paths, adv_paths = list(zip(*advex_matches))

    # noinspection PyTypeChecker
    img_list = eval_class_stability(img_paths, [imgprior], learning_rate, n_iterations, log_freq,
                                    optimizer=optimizer, classifier='alexnet', verbose=True)
    print(img_list)
    np.save(log_path + 'img_log.npy', np.asarray(img_list))

    imgprior.reset()

    # noinspection PyTypeChecker
    adv_list = eval_class_stability(adv_paths, [imgprior], learning_rate, n_iterations, log_freq,
                                    optimizer=optimizer, classifier='alexnet', verbose=True)
    print(adv_list)
    np.save(log_path + 'adv_log.npy', np.asarray(adv_list))


def stability_statistics(path, plot=True, log_freq=None, plot_title=None):
    # log_freq = list(range(1, 5)) + list(range(5, 50, 5)) + list(range(50, 101, 10))

    img_log = np.load(path + 'img_log.npy')
    adv_log = np.load(path + 'adv_log.npy')

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

    # count number of original labels at each time step for img and adv
    src_preserved = 1 - np.minimum(np.abs(img_log.T - src_labels).T, 1)
    count_preserved = np.sum(src_preserved, axis=0)
    print('count of preserved images at each time step', list(count_preserved.astype(np.int)))

    count_restored = np.sum(src_found, axis=0)
    print('count of restored advex at each time step', list(count_restored.astype(np.int)))

    if plot:
        plot_stability_tradeoff(count_preserved, count_restored, log_freq, path, plot_title)


def plot_stability_tradeoff(count_preserved, count_restored, log_freq, path=None, title=None):
    # log_freq = [0] + log_freq

    log_freq = log_freq or range(len(count_restored))

    sns.set_style('darkgrid')
    # sns.set_context('paper')

    plt.figure()
    if title is not None:
        plt.title(title)
    plt.plot(log_freq, count_preserved, 'r-', label='image')
    plt.plot(log_freq, count_restored, 'b-', label='adversarial')
    plt.xlabel('regularization steps')
    plt.ylabel('classified correctly')
    freq = int(max([len(log_freq) // 15, 1]))
    plt.xticks(log_freq[::freq])
    plt.legend()

    if path is not None:
        plt.savefig(path + 'tradeoff.png')
    plt.show()
    plt.close()


def eval_adaptive_forward_opt_dep(image, prior, learning_rate, n_iterations, attack_name, attack_keys, src_label,
                                  classifier='alexnet', verbose=False):
    """
    runs baseline with prior, then adaptive attack
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
        _, logit_tsr, _ = get_classifier_io(classifier, input_init=featmap, input_type='tensor')

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


def adaptive_experiment(learning_rate, n_iterations, attack_name, attack_keys, prior_mode,
                        path, img_log_file, classifier, image_shape, images_file, advex_subdir, verbose):
    """
    constructs adaptive attacks for the prior, records necessary perturbation for oblivious and adaptive attack
    separately records, which inputs are misclassified as result of the regularization alone.
    :param learning_rate:
    :param n_iterations:
    :param attack_name:
    :param attack_keys:
    :param prior_mode:
    :param path:
    :param img_log_file:
    :param classifier:
    :param image_shape:
    :param images_file:
    :param advex_subdir:
    :param verbose:
    :return:
    """

    advex_matches = advex_match_paths(images_file=images_file, advex_subdir=advex_subdir)
    print(advex_matches[0])
    img_log = np.load(path + img_log_file)
    prior = get_default_prior(mode=prior_mode)

    def featmap_transform_fun(x):
        if prior.in_tensor_names != 'image':
            output_name = prior.in_tensor_names
            output_name = output_name[:-2] if output_name.endswith(':0') else output_name
            return AlexNet().build_partial(in_tensor=x, input_name='input', output_name=output_name)
        else:
            return x

    if isinstance(prior, FoEDropoutPrior):
        prior.activate_dropout = False

    with tf.Graph().as_default():
        input_featmap = tf.placeholder(dtype=tf.float32, shape=image_shape)
        featmap, _ = prior.forward_opt_adam(input_featmap, learning_rate, n_iterations,
                                            in_to_loss_featmap_fun=featmap_transform_fun)
        _, logit_tsr, _ = get_classifier_io(classifier, input_init=featmap, input_type='tensor')

        with tf.Session() as sess:
            model = foolbox.models.TensorFlowModel(input_featmap, logit_tsr, bounds=(0, 255))
            criterion = foolbox.criteria.Misclassification()
            attack = get_attack(attack_name, model, criterion)
            if attack_keys is None:
                if attack == 'deepfool':
                    attack_keys = {'steps': 300}
                else:
                    attack_keys = dict()

            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            prior.load_weights(sess)

            noise_norms = []
            src_invariant = []

            adv_path = advex_matches[0][1]
            adaptive_save_path = adv_path.replace('oblivious', 'adaptive_' + prior_mode)
            adaptive_save_dir = '/'.join(adaptive_save_path.split('/')[:-1])
            if not os.path.exists(adaptive_save_dir):
                os.makedirs(adaptive_save_dir)

            for idx, match in enumerate(advex_matches):
                img_path, adv_path = match
                src_label = img_log[idx][0]

                img = load_image(img_path)
                adv = load_image(adv_path)

                oblivious_norm = np.linalg.norm((img - adv).flatten(), ord=2)
                print('oblivious norm', oblivious_norm)

                img_pred = model.predictions(img)
                img_pred_label = np.argmax(img_pred)
                adv_pred = model.predictions(adv)
                adv_pred_label = np.argmax(adv_pred)

                if verbose:
                    noisy_label_name = get_class_name(img_pred_label)
                    if img_pred_label == src_label:
                        print('noisy label {} same as source: {}'.format(img_pred_label, noisy_label_name))
                        src_invariant.append(1)
                        if adv_pred_label == img_pred_label:
                            print('oblivious attack averted')
                        else:

                            print('WARNING: oblivious attack succeeded!')
                    else:
                        print('image with prior misclassified as {}. (label {})'.format(noisy_label_name,
                                                                                        img_pred_label))
                        src_invariant.append(0)
                try:
                    adversarial = attack(image=img, label=img_pred_label, **attack_keys)
                    if adversarial is None:
                        adaptive_norm = np.inf
                        if verbose:
                            print('no adversary found for source label {} using {}'.format(img_pred_label, attack_name))

                    else:
                        fooled_pred = model.predictions(adversarial)
                        fooled_label = np.argmax(fooled_pred)
                        fooled_label_name = get_class_name(fooled_label)
                        adaptive_norm = np.linalg.norm((img - adversarial).flatten(), ord=2)
                        if verbose:
                            print('adversarial image classified as {}. (label {}) '
                                  'Necessary perturbation: {}'.format(fooled_label_name, fooled_label, adaptive_norm))
                        adaptive_save_path = adv_path.replace('oblivious', 'adaptive_' + prior_mode)

                        np.save(adaptive_save_path, adversarial)
                except AssertionError as err:
                    adaptive_norm = -np.inf
                    print('FoolBox failed Assertion: {}'.format(err))

                noise_norms.append((oblivious_norm, adaptive_norm))
                if idx + 1 % 100 == 0:
                    np.save(path + 'noise_norms.npy', np.asarray(noise_norms))
                    np.save(path + 'src_invariants.npy', np.asarray(src_invariant))
        np.save(path + 'noise_norms.npy', np.asarray(noise_norms))
        np.save(path + 'src_invariants.npy', np.asarray(src_invariant))


# noinspection PyTypeChecker
def read_adaptive_log(path):
    noise_norms = np.load(path + 'noise_norms.npy')
    src_invariants = np.load(path + 'src_invariants.npy')

    print('# correct classifiations with prior', np.sum(src_invariants))
    oblivious_norms = noise_norms[:, 0]
    adaptive_norms = noise_norms[:, 1]

    print('# attacks failed within allowed steps', np.sum(adaptive_norms == np.inf))
    print('# errors during attack', np.sum(adaptive_norms == -np.inf))
    success_ids = (adaptive_norms != np.inf) * (adaptive_norms != -np.inf)
    print('# successful attacks', np.sum(success_ids))

    oblivious_norms = oblivious_norms[success_ids]
    adaptive_norms = adaptive_norms[success_ids]
    print('maxmin of adaptive noise', np.max(adaptive_norms), np.min(adaptive_norms))
    diff = adaptive_norms - oblivious_norms
    frac = adaptive_norms / oblivious_norms
    print('meanmedian diff', np.mean(diff), np.median(diff))
    print('meanmedian frac', np.mean(frac), np.median(frac))
    print('# adative noise > oblibious noise', np.sum(diff > 0))
    print('# adative noise < oblibious noise', np.sum(diff < 0))


def noise_norm_histograms(noise_norms):
    pass


def verify_advex_claims(advex_dir='../data/adversarial_examples/foolbox_images/alexnet_val_2k_top1_correct/'
                                  'deepfool_adaptive_dropout_nodrop_train/'):
    advex_files = sorted(os.listdir(advex_dir))
    classifier = 'alexnet'
    n_fooled_correctly = 0
    n_fooled_differently = 0
    n_not_fooled = 0

    with tf.Graph().as_default():

        image_pl, logit_tsr, _ = get_classifier_io(classifier, input_type='placeholder')
        with tf.Session() as sess:
            for file_name in advex_files:
                path = advex_dir + file_name
                print(file_name)
                image_mat = load_image(path, resize=False)
                image_mat = np.expand_dims(image_mat.astype(dtype=np.float32), axis=0)

                pred = sess.run(logit_tsr, feed_dict={image_pl: image_mat})
                current_label = str(np.argmax(pred))
                print('label:', current_label)

                if 't' + current_label in file_name:
                    n_not_fooled += 1
                    print('not fooled')
                elif 'f' + current_label in file_name:
                    n_fooled_correctly += 1
                    print('fooled correctly')
                else:
                    n_fooled_differently += 1
                    print('fooled differently')
        print('not fooled: {}  fooled correctly: {}  fooled differently: {}'.format(n_not_fooled, n_fooled_correctly,
                                                                                    n_fooled_differently))


def ensemble_adaptive_experiment(learning_rate, n_iterations, attack_name, attack_keys, prior_mode,
                                 path, img_log_file, classifier, image_shape, images_file, advex_subdir, verbose,
                                 ensemble_size, target_prob):
        """
        constructs adaptive attacks for the prior, records necessary perturbation for oblivious and adaptive attack
        separately records, which inputs are misclassified as result of the regularization alone.
        :param learning_rate:
        :param n_iterations:
        :param attack_name:
        :param attack_keys:
        :param prior_mode:
        :param path:
        :param img_log_file:
        :param classifier:
        :param image_shape:
        :param images_file:
        :param advex_subdir:
        :param verbose:
        :param ensemble_size:
        :param target_prob:
        :return:
        """

        advex_matches = advex_match_paths(images_file=images_file, advex_subdir=advex_subdir)
        img_log = np.load(path + img_log_file)
        prior = get_default_prior(mode=prior_mode)
        assert isinstance(prior, FoEDropoutPrior)

        with tf.Graph().as_default():
            input_featmap = tf.placeholder(dtype=tf.float32, shape=image_shape)
            masks = prior.make_dropout_masks(ensemble_size, n_iterations)
            featmaps = prior.masked_ensemble_forward_opt_adam(input_featmap, learning_rate, n_iterations, masks)
            _, logit_tsr, _ = get_classifier_io(classifier, input_init=featmaps, input_type='tensor')

            label_var = tf.get_variable('label_var', shape=(), dtype=tf.int32, trainable=False)
            label_pl = tf.placeholder(dtype=tf.int32, shape=())
            label_feed_op = tf.assign(label_var, label_pl)

            logit_tsr = aggregate_ensemble_logits(logit_tsr, label_var, method='default')
            logit_tsr = tf.expand_dims(logit_tsr, axis=0)

            with tf.Session() as sess:
                model = foolbox.models.TensorFlowModel(input_featmap, logit_tsr, bounds=(0, 255))

                criterion = foolbox.criteria.OriginalClassProbability(target_prob)

                attack = get_attack(attack_name, model, criterion)
                if attack_keys is None:
                    if attack == 'deepfool':
                        attack_keys = {'steps': 300}
                    else:
                        attack_keys = dict()

                init_op = tf.global_variables_initializer()
                sess.run(init_op)
                prior.load_weights(sess)

                noise_norms = []
                src_invariant = []

                adv_path = advex_matches[0][1]
                adaptive_save_path = adv_path.replace('oblivious', 'ensemble_adaptive_' + prior_mode)
                adaptive_save_dir = '/'.join(adaptive_save_path.split('/')[:-1])
                if not os.path.exists(adaptive_save_dir):
                    os.makedirs(adaptive_save_dir)

                for idx, match in enumerate(advex_matches):
                    img_path, adv_path = match

                    src_label = img_log[idx][0]
                    sess.run(label_feed_op, feed_dict={label_pl: src_label})

                    img = load_image(img_path)
                    adv = load_image(adv_path)

                    oblivious_norm = np.linalg.norm((img - adv).flatten(), ord=2)
                    print('oblivious norm', oblivious_norm)

                    img_pred = model.predictions(img)
                    img_pred_label = np.argmax(img_pred)
                    adv_pred = model.predictions(adv)
                    adv_pred_label = np.argmax(adv_pred)

                    if verbose:
                        noisy_label_name = get_class_name(img_pred_label)
                        if img_pred_label == src_label:
                            print('noisy label {} same as source: {}'.format(img_pred_label, noisy_label_name))
                            src_invariant.append(1)
                            if adv_pred_label == img_pred_label:
                                print('oblivious attack averted')
                            else:

                                print('WARNING: oblivious attack succeeded!')
                        else:
                            print('image with prior misclassified as {}. (label {})'.format(noisy_label_name,
                                                                                            img_pred_label))
                            src_invariant.append(0)
                    try:
                        adversarial = attack(image=img, label=img_pred_label, **attack_keys)
                        if adversarial is None:
                            adaptive_norm = np.inf
                            if verbose:
                                print('no adversary found for source label {} using {}'.format(img_pred_label,
                                                                                               attack_name))

                        else:
                            fooled_pred = model.predictions(adversarial)
                            fooled_label = np.argmax(fooled_pred)
                            fooled_label_name = get_class_name(fooled_label)
                            adaptive_norm = np.linalg.norm((img - adversarial).flatten(), ord=2)
                            if verbose:
                                print('adversarial image classified as {}. (label {}) '
                                      'Necessary perturbation: {}'.format(fooled_label_name, fooled_label,
                                                                          adaptive_norm))
                            adaptive_save_path = adv_path.replace('oblivious', 'ensemble_adaptive_' + prior_mode)

                            np.save(adaptive_save_path, adversarial)
                    except AssertionError as err:
                        adaptive_norm = -np.inf
                        print('FoolBox failed Assertion: {}'.format(err))

                    noise_norms.append((oblivious_norm, adaptive_norm))
                    if idx + 1 % 100 == 0:
                        np.save('noise_norms.npy', np.asarray(noise_norms))
                        np.save('src_invariants.npy', np.asarray(src_invariant))
            np.save('noise_norms.npy', np.asarray(noise_norms))
            np.save('src_invariants.npy', np.asarray(src_invariant))


def aggregate_ensemble_logits(logits, label, method):
    lse = tf.reduce_logsumexp(logits, axis=1)
    scaled_logits = tf.transpose(tf.transpose(logits) - lse)
    if method == 'mean':
        return tf.reduce_mean(scaled_logits, axis=0)
    elif method == 'softmax':
        mean_logits = tf.reduce_mean(scaled_logits, axis=0)
        label_logits = scaled_logits[:, label]
        softmax_logits = tf.nn.softmax(label_logits)
        weighted_avg = tf.reduce_sum(softmax_logits * label_logits)
        splits = tf.stack([label, 1, 999 - label])
        split_mean_logits = tf.split(mean_logits, splits, axis=0)
        merged_logits = tf.stack([split_mean_logits[0], weighted_avg, split_mean_logits[2]], axis=0)
        return merged_logits


def compare_adams(advex_dir, prior_mode='dropout_nodrop_train', learning_rate=0.1, n_iterations=2):
    """
    compares gradients and updates of rolled out and built in adam optimizer (potentially also accumulator values)
    """

    advex_files = sorted(os.listdir(advex_dir))
    iterative_prior = get_default_prior(prior_mode)
    iterative_prior.name = 'IterativePrior'
    rollout_prior = get_default_prior(prior_mode)
    rollout_prior.name = 'RollooutPrior'
    image_shape = (1, 227, 227, 3)

    with tf.Graph().as_default():
        image_var = tf.get_variable('advex', shape=image_shape, dtype=tf.float32)
        image_regularized, _ = rollout_prior.forward_opt_adam(image_var, learning_rate, n_iterations)

        iterative_prior.build(featmap_tensor=image_var)
        iterative_loss = iterative_prior.get_loss()
        adam_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grad_var_pairs = adam_opt.compute_gradients(iterative_loss, [image_var])
        iterative_image_grad = grad_var_pairs[0][0]
        iterative_image_grad = tf.Print(iterative_image_grad, [iterative_image_grad],
                                        message='iterative_grad:', summarize=10)
        train_op = adam_opt.apply_gradients(grad_var_pairs)

        image_pl = tf.placeholder(dtype=tf.float32, shape=image_shape)
        feed_op = tf.assign(image_var, image_pl)
        glob_vars = tf.global_variables()
        print([v.name for v in glob_vars])
        init_op = tf.variables_initializer(glob_vars)

        with tf.Session() as sess:
            for file_name in advex_files[:5]:

                sess.run(init_op)
                iterative_prior.load_weights(sess)
                rollout_prior.load_weights(sess)

                path = advex_dir + file_name
                print(file_name)
                image_mat = load_image(path, resize=False)
                image_mat = np.expand_dims(image_mat.astype(dtype=np.float32), axis=0)
                sess.run(feed_op, feed_dict={image_pl: image_mat})
                rollout_reg = sess.run(image_regularized)

                iterative_grads = []
                for count in range(n_iterations):
                    grad, _ = sess.run([iterative_image_grad, train_op])
                    iterative_grads.append(grad)
                iterative_reg = sess.run(image_var)

                rollout_diff = image_mat - rollout_reg
                iterative_diff = image_mat - iterative_reg
                rollout_norm = np.linalg.norm(rollout_diff)
                iterative_norm = np.linalg.norm(iterative_diff)
                print('diff norms: rollout {}  iterative {}'.format(rollout_norm, iterative_norm))
