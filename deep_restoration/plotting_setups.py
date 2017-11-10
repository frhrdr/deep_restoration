from utils.tb_log_readers import plot_opt_inv_experiment


def plot_example_exp():
    path = '../logs/opt_inversion/alexnet/slim_vs_img/c2l_to_c1l'
    exp_subdirs = {'No prior': 'no_prior',
                   'Pre-image with prior': 'pre_image_8x8_full_prior/1e-3',
                   'Pre-image with no prior': 'pre_image_no_prior'}
    log_tags = {'Total loss': 'Total_Loss',
                'Reconstruction error': 'MSE_Reconstruction_1'}
    plot_opt_inv_experiment(path, exp_subdirs, log_tags)


def plot_c1l_prior_comp():
    path = '../logs/opt_inversion/alexnet/old/c2l_to_c1l_select'
    exp_subdirs = {'No prior': 'no_prior/adam', 'Total variation prior': 'total_variation_prior',
                   '3x3 patch prior': '3x3_full_prior', '3x3 patch prior + 8x8 channel': '3x3_full_plus_8x8_chan_prior',
                   '8x8 channel prior': '8x8_chan_prior', '8x8 patch prior': '8x8_full_prior'}
    log_tags = {'Total loss': 'Total_Loss',
                'Reconstruction error': 'MSE_Reconstruction_1'}
    plot_opt_inv_experiment(path, exp_subdirs, log_tags, log_subdir='', max_steps=500)

