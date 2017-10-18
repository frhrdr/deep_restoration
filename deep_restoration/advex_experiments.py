from utils.foolbox_utils import get_default_prior, stability_experiment, adaptive_experiment


def stability_experiment_fullprior(images_file='alexnet_val_2k_top1_correct.txt',
                                   advex_subdir='alexnet_val_2k_top1_correct/deepfool_oblivious/'):
    imgprior = get_default_prior(mode='full')
    optimizer = 'adam'
    # learning_rate = 1e-0
    # n_iterations = 100
    # log_freq = list(range(1, 5)) + list(range(5, 50, 5)) + list(range(50, 101, 10))
    learning_rate = 1e-1
    n_iterations = 30
    log_freq = 1
    # noinspection PyTypeChecker
    stability_experiment(images_file=images_file, advex_subdir=advex_subdir, imgprior=imgprior,
                         optimizer=optimizer, learning_rate=learning_rate, n_iterations=n_iterations, log_freq=log_freq)


def stability_experiment_dropoutprior(images_file='alexnet_val_2k_top1_correct.txt',
                                      advex_subdir='alexnet_val_2k_top1_correct/deepfool_oblivious/',
                                      nodrop_train=False):
    if nodrop_train:
        imgprior = get_default_prior(mode='dropout_nodrop_train')
    else:
        imgprior = get_default_prior(mode='dropout')
    optimizer = 'adam'
    learning_rate = 1e-1
    n_iterations = 30
    log_freq = 1
    imgprior.activate_dropout = False
    log_path = '../logs/adversarial_examples/alexnet_top1/deepfool/oblivious_dropoutprior_dodrop_train/'
    # noinspection PyTypeChecker
    stability_experiment(images_file=images_file, advex_subdir=advex_subdir, imgprior=imgprior,
                         optimizer=optimizer, learning_rate=learning_rate, n_iterations=n_iterations, log_freq=log_freq,
                         log_path=log_path)


def stability_experiment_dodrop_adaptive(images_file='alexnet_val_2k_top1_correct.txt',
                                         advex_subdir='alexnet_val_2k_top1_correct/'
                                                      'deepfool_adaptive_dropout_nodrop_train/'):
    imgprior = get_default_prior(mode='dropout_nodrop_train')
    optimizer = 'adam'
    learning_rate = 1e-1
    n_iterations = 30
    log_freq = 1
    log_path = '../logs/adversarial_examples/alexnet_top1/deepfool/adaptive_dropoutprior_nodrop_train/dodrop_test/'
    # noinspection PyTypeChecker
    stability_experiment(images_file=images_file, advex_subdir=advex_subdir, imgprior=imgprior,
                         optimizer=optimizer, learning_rate=learning_rate, n_iterations=n_iterations, log_freq=log_freq,
                         log_path=log_path)


def stability_experiment_dropoutprior_nodrop_train_100(images_file='subset_100_images.txt',
                                                       advex_subdir='100_dataset/deepfool_oblivious/',
                                                       nodrop_train=True):
    if nodrop_train:
        imgprior = get_default_prior(mode='dropout_nodrop_train')
    else:
        imgprior = get_default_prior(mode='dropout')
    optimizer = 'adam'
    learning_rate = 1e-1
    n_iterations = 30
    log_freq = 1
    imgprior.activate_dropout = False
    log_path = '../logs/adversarial_examples/100_dataset/deepfool/oblivious_dropoutprior_nodrop_train/'
    # noinspection PyTypeChecker
    stability_experiment(images_file=images_file, advex_subdir=advex_subdir, imgprior=imgprior,
                         optimizer=optimizer, learning_rate=learning_rate, n_iterations=n_iterations, log_freq=log_freq,
                         log_path=log_path)


def stability_experiment_dodrop_adaptive_100(images_file='subset_100_images.txt',
                                             advex_subdir='100_dataset/deepfool_adaptive_dropout_nodrop_train/'):
    imgprior = get_default_prior(mode='dropout_nodrop_train')
    optimizer = 'adam'
    learning_rate = 1e-1
    n_iterations = 30
    log_freq = 1
    log_path = '../logs/adversarial_examples/100_dataset/deepfool/adaptive_dropoutprior_nodrop_train/dodrop_test/'
    # noinspection PyTypeChecker
    stability_experiment(images_file=images_file, advex_subdir=advex_subdir, imgprior=imgprior,
                         optimizer=optimizer, learning_rate=learning_rate, n_iterations=n_iterations, log_freq=log_freq,
                         log_path=log_path)


def stability_experiment_nodrop_adaptive(images_file='alexnet_val_2k_top1_correct.txt',
                                         advex_subdir='alexnet_val_2k_top1_correct/'
                                                      'deepfool_adaptive_dropout_nodrop_train/'):
    imgprior = get_default_prior(mode='dropout_nodrop_train')
    optimizer = 'adam'
    learning_rate = 1e-1
    n_iterations = 30
    log_freq = 1
    imgprior.activate_dropout = False
    log_path = '../logs/adversarial_examples/alexnet_top1/deepfool/adaptive_dropoutprior_nodrop_train/nodrop_test/'
    # noinspection PyTypeChecker
    stability_experiment(images_file=images_file, advex_subdir=advex_subdir, imgprior=imgprior,
                         optimizer=optimizer, learning_rate=learning_rate, n_iterations=n_iterations, log_freq=log_freq,
                         log_path=log_path)


def adaptive_experiment_200(learning_rate=0.1, n_iterations=5, attack_name='deepfool', attack_keys=None, verbose=True):
    path = '../logs/adversarial_examples/deepfool_oblivious_198/'
    img_log_file = 'img_log_198_fine.npy'
    classifier = 'alexnet'
    image_shape = (1, 227, 227, 3)
    images_file = 'subset_cutoff_200_images.txt'
    advex_subdir = '200_dataset/deepfool_oblivious/'
    prior_mode = 'full'
    adaptive_experiment(learning_rate=learning_rate, n_iterations=n_iterations, attack_name=attack_name,
                        attack_keys=attack_keys, prior_mode=prior_mode, path=path, img_log_file=img_log_file,
                        classifier=classifier,
                        image_shape=image_shape, images_file=images_file, advex_subdir=advex_subdir,
                        verbose=verbose)


def adaptive_experiment_alex_top1(learning_rate=0.1, n_iterations=5, attack_name='deepfool',
                                  attack_keys=None, verbose=True):

    path = '../logs/adversarial_examples/alexnet_top1/deepfool/oblivious_fullprior/'
    img_log_file = 'img_log.npy'
    classifier = 'alexnet'
    image_shape = (1, 227, 227, 3)
    images_file = 'alexnet_val_2k_top1_correct.txt'
    advex_subdir = 'alexnet_val_2k_top1_correct/deepfool_oblivious/'
    prior_mode = 'full'
    adaptive_experiment(learning_rate=learning_rate, n_iterations=n_iterations, attack_name=attack_name,
                        attack_keys=attack_keys, prior_mode=prior_mode, path=path, img_log_file=img_log_file,
                        classifier=classifier,
                        image_shape=image_shape, images_file=images_file, advex_subdir=advex_subdir,
                        verbose=verbose)


def adaptive_experiment_alex_top1_dropout_prior_nodrop_train(learning_rate=0.1, n_iterations=5, attack_name='deepfool',
                                                             attack_keys=None, verbose=True):

    path = '../logs/adversarial_examples/alexnet_top1/deepfool/oblivious_dropoutprior_nodrop_train/'
    img_log_file = 'img_log.npy'
    classifier = 'alexnet'
    image_shape = (1, 227, 227, 3)
    images_file = 'alexnet_val_2k_top1_correct.txt'
    advex_subdir = 'alexnet_val_2k_top1_correct/deepfool_oblivious/'
    prior_mode = 'dropout_nodrop_train'
    deactivate_dropout = True
    adaptive_experiment(learning_rate=learning_rate, n_iterations=n_iterations, attack_name=attack_name,
                        attack_keys=attack_keys, prior_mode=prior_mode, path=path, img_log_file=img_log_file,
                        classifier=classifier,
                        image_shape=image_shape, images_file=images_file, advex_subdir=advex_subdir,
                        verbose=verbose, deactivate_dropout=deactivate_dropout)


def adaptive_experiment_alex_top1_dropout_prior_dodrop_train(learning_rate=0.1, n_iterations=5, attack_name='deepfool',
                                                             attack_keys=None, verbose=True):

    path = '../logs/adversarial_examples/alexnet_top1/deepfool/oblivious_dropoutprior_dodrop_train/'
    img_log_file = 'img_log.npy'
    classifier = 'alexnet'
    image_shape = (1, 227, 227, 3)
    images_file = 'alexnet_val_2k_top1_correct.txt'
    advex_subdir = 'alexnet_val_2k_top1_correct/deepfool_oblivious/'
    prior_mode = 'dropout'
    deactivate_dropout = True
    adaptive_experiment(learning_rate=learning_rate, n_iterations=n_iterations, attack_name=attack_name,
                        attack_keys=attack_keys, prior_mode=prior_mode, path=path, img_log_file=img_log_file,
                        classifier=classifier,
                        image_shape=image_shape, images_file=images_file, advex_subdir=advex_subdir,
                        verbose=verbose, deactivate_dropout=deactivate_dropout)


def adaptive_experiment_100_dropout_prior_nodrop_train(learning_rate=0.1, n_iterations=5, attack_name='deepfool',
                                                       attack_keys=None, verbose=True):

    path = '../logs/adversarial_examples/100_dataset/deepfool/oblivious_dropoutprior_nodrop_train/'
    img_log_file = 'img_log.npy'
    classifier = 'alexnet'
    image_shape = (1, 227, 227, 3)
    images_file = 'subset_100_images.txt'
    advex_subdir = '100_dataset/deepfool_oblivious/'
    prior_mode = 'dropout_nodrop_train'
    deactivate_dropout = True
    adaptive_experiment(learning_rate=learning_rate, n_iterations=n_iterations, attack_name=attack_name,
                        attack_keys=attack_keys, prior_mode=prior_mode, path=path, img_log_file=img_log_file,
                        classifier=classifier,
                        image_shape=image_shape, images_file=images_file, advex_subdir=advex_subdir,
                        verbose=verbose, deactivate_dropout=deactivate_dropout)
