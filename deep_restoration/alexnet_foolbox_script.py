import numpy as np
import tensorflow as tf
from utils.foolbox_utils import make_targeted_examples, make_small_untargeted_dataset, get_prior_scores_per_image, \
    compare_images_to_untargeted_adv_ex, eval_class_stability, stability_statistics, \
    make_untargeted_dataset, read_adaptive_log, \
    verify_advex_claims, compare_adams
from utils.mean_filter_benchmark import mean_filter_benchmark, mean_log_statistics, mean_adaptive_attacks_200
from advex_experiments import adaptive_experiment_alex_top1_dropout_prior_dodrop_train, \
    stability_experiment_dropoutprior, ensemble_adaptive_experiment_100_dropout_prior_nodrop_train, \
    stability_experiment_nodrop_adaptive, stability_experiment_dropoutprior, adaptive_experiment_alex_top1, \
    stability_experiment_fullprior, adaptive_experiment_alex_top1_dropout_prior_dodrop_train, \
    adaptive_experiment_alex_top1_dropout_prior_nodrop_train, c1l_prior_stability_experiment, \
    c1l_prior_adaptive_experiment, stability_experiment_fullprior_adaptive
from utils.mean_filter_benchmark import default_mean_filter_exp
from modules.foe_full_prior import FoEFullPrior

# image_dir = '../data/adversarial_examples/foolbox_images/small_dataset/lbfgs/'
# image_names = ['val53_t844_f39.bmp',
#                'val53_t844_f844.bmp',
#                'val53_t844_f970.bmp',
#                'val76_t860_f424.bmp',
#                'val81_t970_f970.bmp',
#                'val99_t949_f934.bmp',
#                'val99_t949_f949.bmp',
#                'val99_t949_f970.bmp',
#                'val106_t824_f906.bmp',
#                'val108_t889_f486.bmp'
#                ]
# image_dir = '../data/selected/images_resized_227/val{}.bmp'
# image_names = [53, 76, 81, 99, 106, 108, 129, 153, 157, 160]
#
# image_paths = [image_dir.format(k) for k in image_names]
#
# make_small_untargeted_dataset()

# imgprior = FoEFullPrior('rgb_scaled:0', 1e-5, 'alexnet', [12, 12], 1.0, n_components=1000, n_channels=3,
#                         n_features_white=12**2*3, dist='student', mean_mode='gc', sdev_mode='gc', whiten_mode='pca',
#                         name=None, load_name=None, dir_name=None, load_tensor_names='image')

# get_prior_scores_per_image(image_paths, [imgprior])

# compare_images_to_untargeted_adv_ex([imgprior])

# TESTING REGULARIZER BEHAVIOUR ########################################################################################
# imgprior = FoEFullPrior('rgb_scaled:0', 1e-5, 'alexnet', [8, 8], 1.0, n_components=512, n_channels=3,
#                         n_features_white=8**2*3-1, dist='student', mean_mode='gc', sdev_mode='gc', whiten_mode='pca',
#                         name=None, load_name='FoEPrior', dir_name=None, load_tensor_names='image')
#
# # image_file = '../data/selected/images_resized_227/val53.bmp'
# # image_file = '../data/adversarial_examples/foolbox_images/small_dataset/lbfgs/val76_t860_f424.bmp'
# image_file = '../data/adversarial_examples/foolbox_images/200_dataset/deepfool/ILSVRC2012_val_00000053_t844_f530.npy'
# priors = [imgprior]
# learning_rate = 1e-0
# n_iterations = 100
# log_freq = 5
# log_list = eval_class_stability(image_file, priors, learning_rate, n_iterations, log_freq,
#                                 optimizer='adam', classifier='alexnet', verbose=True)
# print(log_list)
# stability_experiment_fullprior()

# path = '../logs/adversarial_examples/alexnet_top1/deepfool/oblivious_fullprior/'
# path = '../logs/adversarial_examples/alexnet_top1/deepfool/adaptive_dropoutprior_nodrop_train1024/dodrop_test/lr04/'
# path = '../logs/adversarial_examples/alexnet_top1/deepfool/oblivious_dropoutprior_nodrop_train1024/lr06/'
# path = '../logs/adversarial_examples/alexnet_top1/deepfool/oblivious_fullc1l6000/lr06/'
# stability_statistics(path, plot=True)
# adaptive_experiment_alex_top1()

# make_untargeted_dataset(image_subset='alexnet_val_2k_top1_correct.txt',
#                         attack_name='gradientsign', attack_keys=None)
# stability_experiment_fullprior(images_file='alexnet_val_2k_top1_correct.txt',
#                                advex_subdir='alexnet_val_2k_top1_correct/gradientsign_oblivious/',
#                                attack_name='gradientsign')

# read_adaptive_log('../logs/adversarial_examples/alexnet_top1/deepfool/adaptive_fullprior/')

# adaptive_experiment_alex_top1_dropout_prior_dodrop_train()

# read_adaptive_log('../logs/adversarial_examples/alexnet_top1/deepfool/oblivious_fullc1l6000/lr06/')
# stability_experiment_dodrop_adaptive()
# verify_advex_claims()
# stability_experiment_nodrop_adaptive()
# stability_experiment_dropoutprior()
# adaptive_experiment_alex_top1_dropout_prior_dodrop_train()

# path = '../data/adversarial_examples/foolbox_images/100_dataset/deepfool_adaptive_dropout_nodrop_train/'
# compare_adams(path, n_iterations=5)

# ensemble_adaptive_experiment_100_dropout_prior_nodrop_train()

# stability_experiment_nodrop_adaptive()
# stability_experiment_dropoutprior(nodrop_train=True)

# default_mean_filter_exp()
# adaptive_experiment_alex_top1(attack_name='gradientsign')
# adaptive_experiment_alex_top1_dropout_prior_nodrop_train(attack_name='gradientsign')
# adaptive_experiment_alex_top1_dropout_prior_dodrop_train(attack_name='gradientsign')
# c1l_prior_stability_experiment()
# c1l_prior_adaptive_experiment()
# stability_experiment_fullprior_adaptive()
stability_statistics('../logs/adversarial_examples/alexnet_top1/deepfool/adaptive_fullprior512/lr06/',
                     plot_title='adaptive deepfool - FoE prior 512 components')
