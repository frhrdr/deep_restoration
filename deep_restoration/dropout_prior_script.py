from modules.foe_dropout_prior import FoEDropoutPrior

imgprior = FoEDropoutPrior('rgb_scaled:0', 1e-5, 'alexnet', [8, 8], 1.0, n_components=1024, n_channels=3,
                           n_features_white=8 ** 2 * 3 - 1, dist='student', mean_mode='gc', sdev_mode='gc',
                           whiten_mode='pca',
                           activate_dropout=True, make_switch=False, dropout_prob=0.5)

imgprior.train_prior(batch_size=250, n_iterations=25000, lr=3e-5,
                     lr_lower_points=((0, 1e-0), (10000, 1e-1),
                                      (13000, 3e-2),
                                      (15000, 1e-2), (17000, 3e-3), (19000, 1e-3),
                                      (21000, 1e-4), (23000, 3e-5)),
                     grad_clip=1e-3,
                     n_data_samples=100000, n_val_samples=500,
                     log_freq=5000, summary_freq=10, print_freq=100,
                     prev_ckpt=0,
                     optimizer_name='adam', plot_filters=True, stop_on_overfit=False)
