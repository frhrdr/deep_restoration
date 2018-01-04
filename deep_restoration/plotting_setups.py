from utils.tb_log_readers import plot_opt_inv_experiment
import numpy as np


def plot_example_exp():
    path = '../logs/opt_inversion/alexnet/slim_vs_img/c2l_to_c1l'
    exp_subdirs = {'No prior': 'no_prior',
                   'Pre-image with prior': 'pre_image_8x8_full_prior/1e-3',
                   'Pre-image with no prior': 'pre_image_no_prior'}
    log_tags = {'Total loss': 'Total_Loss',
                'Reconstruction error': 'MSE_Reconstruction_1'}
    plot_opt_inv_experiment(path, exp_subdirs, log_tags)


def plot_c1l_prior_comp():
    path = '../logs/opt_inversion/alexnet/c2l_to_c1l/'
    exp_subdirs = {'No prior': 'no_prior/adam/', 'Total variation prior': 'tv_prior/run_final/',
                   '3x3 patch prior': 'slim_prior/run_final/', '3x3 patch prior + 8x8 channel': 'dual_prior/run_final/',
                   '8x8 channel prior': 'chan_prior/adam/run_final/', '8x8 patch prior': 'full_prior/run_final/'}
    log_tags = {'Total loss': 'Total_Loss',
                'Reconstruction error': 'MSE_Reconstruction_1',
                'Matching error': 'MSE_conv2_1'}
    plot_opt_inv_experiment(path, exp_subdirs, log_tags, log_subdir='summaries')


def vgg_rec_collage():
    path = '../logs/opt_inversion/vgg16/image_rec/{}/val{}/full512/mats/rec_{}.png'

    layer_choices = [('pool1', '25000'), ('pool2', '25000'), ('pool3', '25000'), ('pool4', '25000'), ('pool5', '25000'),
                     ('fc6/lin', '17000'), ('fc7/lin', '17000'), ('fc8/lin', '17000')]

    img_numbers = [53, 78, 81, 99, 106, 108, 129, 153, 157, 160]

    cols = []
    for l, r in layer_choices:
        col = []
        for n in img_numbers:
            mat = np.load(path.format(l, n, r))
            col.append(mat)
        cols.append(np.concatenate(col, axis=0))
    img = np.concatenate(cols, axis=1)
    print(img.shape)