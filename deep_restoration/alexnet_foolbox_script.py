import numpy as np
import tensorflow as tf
from utils.foolbox_utils import make_targeted_examples, make_small_untargeted_dataset, get_prior_scores_per_image, compare_images_to_untargeted_adv_ex
from modules.foe_full_prior import FoEFullPrior

imgprior = FoEFullPrior('rgb_scaled:0', 1e-5, 'alexnet', [12, 12], 1.0, n_components=1000, n_channels=3,
                        n_features_white=12**2*3, dist='student', mean_mode='gc', sdev_mode='gc', whiten_mode='pca',
                        name=None, load_name=None, dir_name=None, load_tensor_names='image')

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
# get_prior_scores_per_image(image_paths, [imgprior])

# make_small_untargeted_dataset()
compare_images_to_untargeted_adv_ex([imgprior])