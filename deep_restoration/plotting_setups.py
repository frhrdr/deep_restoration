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
    path = '../logs/opt_inversion/alexnet/c2l_to_c1l/'
    exp_subdirs = {'No prior': 'no_prior/adam/', 'Total variation prior': 'tv_prior/run_final/',
                   '3x3 patch prior': 'slim_prior/run_final/', '3x3 patch prior + 8x8 channel': 'dual_prior/run_final/',
                   '8x8 channel prior': 'chan_prior/adam/run_final/', '8x8 patch prior': 'full_prior/run_final/'}
    log_tags = {'Total loss': 'Total_Loss',
                'Reconstruction error': 'MSE_Reconstruction_1',
                'Matching error': 'MSE_conv2_1'}
    plot_opt_inv_experiment(path, exp_subdirs, log_tags, log_subdir='summaries')

