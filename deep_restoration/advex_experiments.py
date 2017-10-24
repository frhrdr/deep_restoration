from utils.foolbox_utils import stability_experiment, adaptive_experiment, \
    ensemble_adaptive_experiment
from utils.default_priors import get_default_prior


def stability_experiment_fullprior(images_file='alexnet_val_2k_top1_correct.txt',
                                   advex_subdir='alexnet_val_2k_top1_correct/deepfool_oblivious/',
                                   attack_name='deepfool'):
    imgprior = get_default_prior(mode='full512')
    optimizer = 'adam'
    # learning_rate = 1e-0
    # n_iterations = 100
    # log_freq = list(range(1, 5)) + list(range(5, 50, 5)) + list(range(50, 101, 10))
    learning_rate = 10.
    n_iterations = 30
    log_freq = 1
    log_path = '../logs/adversarial_examples/alexnet_top1/{}/oblivious_fullprior512/lr10/'.format(attack_name)
    # noinspection PyTypeChecker
    stability_experiment(images_file=images_file, advex_subdir=advex_subdir, imgprior=imgprior,
                         optimizer=optimizer, learning_rate=learning_rate, n_iterations=n_iterations, log_freq=log_freq,
                         log_path=log_path)


def stability_experiment_dropoutprior(images_file='alexnet_val_2k_top1_correct.txt',
                                      advex_subdir='alexnet_val_2k_top1_correct/deepfool_oblivious/',
                                      attack_name='deepfool',
                                      nodrop_train=False):
    if nodrop_train:
        imgprior = get_default_prior(mode='dropout_nodrop_train1024')
    else:
        imgprior = get_default_prior(mode='dropout1024')
    optimizer = 'adam'
    learning_rate = 0.6
    n_iterations = 10
    log_freq = 1
    imgprior.activate_dropout = False
    log_path = '../logs/adversarial_examples/alexnet_top1/{}/oblivious_dropoutprior_nodrop_train1024/lr06/'.format(attack_name)
    # noinspection PyTypeChecker
    stability_experiment(images_file=images_file, advex_subdir=advex_subdir, imgprior=imgprior,
                         optimizer=optimizer, learning_rate=learning_rate, n_iterations=n_iterations, log_freq=log_freq,
                         log_path=log_path)


def stability_experiment_dodrop_adaptive(images_file='alexnet_val_2k_top1_correct.txt',
                                         advex_subdir='alexnet_val_2k_top1_correct/'
                                                      'deepfool_adaptive_dropout_nodrop_train1024/',
                                         attack_name='deepfool'):
    imgprior = get_default_prior(mode='dropout_nodrop_train1024')
    optimizer = 'adam'
    learning_rate = 0.6
    n_iterations = 10
    log_freq = 1
    log_path = '../logs/adversarial_examples/alexnet_top1/{}/adaptive_dropoutprior_nodrop_train1024/dodrop_test/lr06/'.format(attack_name)
    # noinspection PyTypeChecker
    stability_experiment(images_file=images_file, advex_subdir=advex_subdir, imgprior=imgprior,
                         optimizer=optimizer, learning_rate=learning_rate, n_iterations=n_iterations, log_freq=log_freq,
                         log_path=log_path)


def stability_experiment_nodrop_adaptive(images_file='alexnet_val_2k_top1_correct.txt',
                                         advex_subdir='alexnet_val_2k_top1_correct/'
                                                      'deepfool_adaptive_dropout_nodrop_train1024/'):
    imgprior = get_default_prior(mode='dropout_nodrop_train1024')
    optimizer = 'adam'
    learning_rate = 0.6
    n_iterations = 10
    log_freq = 1
    imgprior.activate_dropout = False
    log_path = '../logs/adversarial_examples/alexnet_top1/deepfool/adaptive_dropoutprior_nodrop_train1024/nodrop_test/lr06'
    # noinspection PyTypeChecker
    stability_experiment(images_file=images_file, advex_subdir=advex_subdir, imgprior=imgprior,
                         optimizer=optimizer, learning_rate=learning_rate, n_iterations=n_iterations, log_freq=log_freq,
                         log_path=log_path)


def adaptive_experiment_alex_top1(learning_rate=0.6, n_iterations=5, attack_name='deepfool',
                                  attack_keys=None, verbose=True):

    path = '../logs/adversarial_examples/alexnet_top1/{}/oblivious_fullprior512/lr06/'.format(attack_name)
    img_log_file = 'img_log.npy'
    classifier = 'alexnet'
    image_shape = (1, 227, 227, 3)
    images_file = 'alexnet_val_2k_top1_correct.txt'
    advex_subdir = 'alexnet_val_2k_top1_correct/{}_oblivious/'.format(attack_name)
    prior_mode = 'full512'
    adaptive_experiment(learning_rate=learning_rate, n_iterations=n_iterations, attack_name=attack_name,
                        attack_keys=attack_keys, prior_mode=prior_mode, path=path, img_log_file=img_log_file,
                        classifier=classifier,
                        image_shape=image_shape, images_file=images_file, advex_subdir=advex_subdir,
                        verbose=verbose)


def adaptive_experiment_alex_top1_dropout_prior_nodrop_train(learning_rate=0.6, n_iterations=5, attack_name='deepfool',
                                                             attack_keys=None, verbose=True):

    path = '../logs/adversarial_examples/alexnet_top1/{}/oblivious_dropoutprior_nodrop_train/'.format(attack_name)
    img_log_file = 'img_log.npy'
    classifier = 'alexnet'
    image_shape = (1, 227, 227, 3)
    images_file = 'alexnet_val_2k_top1_correct.txt'
    advex_subdir = 'alexnet_val_2k_top1_correct/{}_oblivious/'.format(attack_name)
    prior_mode = 'dropout_nodrop_train1024'
    adaptive_experiment(learning_rate=learning_rate, n_iterations=n_iterations, attack_name=attack_name,
                        attack_keys=attack_keys, prior_mode=prior_mode, path=path, img_log_file=img_log_file,
                        classifier=classifier,
                        image_shape=image_shape, images_file=images_file, advex_subdir=advex_subdir,
                        verbose=verbose)


def adaptive_experiment_alex_top1_dropout_prior_dodrop_train(learning_rate=0.6, n_iterations=5, attack_name='deepfool',
                                                             attack_keys=None, verbose=True):

    path = '../logs/adversarial_examples/alexnet_top1/{}/oblivious_dropoutprior_dodrop_train/'.format(attack_name)
    img_log_file = 'img_log.npy'
    classifier = 'alexnet'
    image_shape = (1, 227, 227, 3)
    images_file = 'alexnet_val_2k_top1_correct.txt'
    advex_subdir = 'alexnet_val_2k_top1_correct/{}_oblivious/'.format(attack_name)
    prior_mode = 'dropout1024'
    adaptive_experiment(learning_rate=learning_rate, n_iterations=n_iterations, attack_name=attack_name,
                        attack_keys=attack_keys, prior_mode=prior_mode, path=path, img_log_file=img_log_file,
                        classifier=classifier,
                        image_shape=image_shape, images_file=images_file, advex_subdir=advex_subdir,
                        verbose=verbose)


def adaptive_experiment_100_dropout_prior_nodrop_train(learning_rate=0.1, n_iterations=5, attack_name='deepfool',
                                                       attack_keys=None, verbose=True):

    path = '../logs/adversarial_examples/100_dataset/deepfool/oblivious_dropoutprior_nodrop_train/'
    img_log_file = 'img_log.npy'
    classifier = 'alexnet'
    image_shape = (1, 227, 227, 3)
    images_file = 'subset_100_images.txt'
    advex_subdir = '100_dataset/deepfool_oblivious/'
    prior_mode = 'dropout_nodrop_train'
    adaptive_experiment(learning_rate=learning_rate, n_iterations=n_iterations, attack_name=attack_name,
                        attack_keys=attack_keys, prior_mode=prior_mode, path=path, img_log_file=img_log_file,
                        classifier=classifier,
                        image_shape=image_shape, images_file=images_file, advex_subdir=advex_subdir,
                        verbose=verbose)


def ensemble_adaptive_experiment_100_dropout_prior_nodrop_train(learning_rate=0.7, n_iterations=1,
                                                                attack_name='deepfool', attack_keys=None, verbose=True):

    path = '../logs/adversarial_examples/100_dataset/deepfool/oblivious_dropoutprior_nodrop_train/'
    img_log_file = 'img_log.npy'
    classifier = 'alexnet'
    image_shape = (1, 227, 227, 3)
    images_file = 'subset_100_images.txt'
    advex_subdir = '100_dataset/deepfool_oblivious/'
    prior_mode = 'dropout_nodrop_train512'
    ensemble_size = 10
    ensemble_adaptive_experiment(learning_rate=learning_rate, n_iterations=n_iterations, attack_name=attack_name,
                                 attack_keys=attack_keys, prior_mode=prior_mode, path=path, img_log_file=img_log_file,
                                 classifier=classifier,
                                 image_shape=image_shape, images_file=images_file, advex_subdir=advex_subdir,
                                 verbose=verbose, ensemble_size=ensemble_size)


def c1l_prior_stability_experiment(images_file='alexnet_val_2k_top1_correct.txt',
                                   advex_subdir='alexnet_val_2k_top1_correct/deepfool_oblivious/',
                                   attack_name='deepfool', prior_mode='fullc1l6000'):
    imgprior = get_default_prior(mode=prior_mode)
    optimizer = 'adam'
    learning_rate = 0.6
    n_iterations = 30
    log_freq = 1
    log_path = '../logs/adversarial_examples/alexnet_top1/{}/oblivious_{}/lr06/'.format(attack_name, prior_mode)
    # noinspection PyTypeChecker
    stability_experiment(images_file=images_file, advex_subdir=advex_subdir, imgprior=imgprior,
                         optimizer=optimizer, learning_rate=learning_rate, n_iterations=n_iterations, log_freq=log_freq,
                         log_path=log_path)


def c1l_prior_adaptive_experiment(learning_rate=0.6, n_iterations=2, attack_name='deepfool',
                                  attack_keys=None, verbose=True, prior_mode='fullc1l6000'):

    path = '../logs/adversarial_examples/alexnet_top1/{}/oblivious_{}/lr06/'.format(attack_name, prior_mode)
    img_log_file = 'img_log.npy'
    classifier = 'alexnet'
    image_shape = (1, 227, 227, 3)
    images_file = 'alexnet_val_2k_top1_correct.txt'
    advex_subdir = 'alexnet_val_2k_top1_correct/{}_oblivious/'.format(attack_name)
    adaptive_experiment(learning_rate=learning_rate, n_iterations=n_iterations, attack_name=attack_name,
                        attack_keys=attack_keys, prior_mode=prior_mode, path=path, img_log_file=img_log_file,
                        classifier=classifier,
                        image_shape=image_shape, images_file=images_file, advex_subdir=advex_subdir,
                        verbose=verbose)
