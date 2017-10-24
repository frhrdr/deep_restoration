from modules.foe_dropout_prior import FoEDropoutPrior
from modules.foe_full_prior import FoEFullPrior


def get_default_prior(mode, custom_weighting=None):
    assert mode in ('full512', 'dropout1024', 'dropout_nodrop_train1024', 'dropout_nodrop_train512', 'fullc1l6000',
                    'slimc5l5000', 'full512logistic')
    if mode == 'full512':
        p = FoEFullPrior('pre_featmap:0', 1e-5, 'alexnet', [8, 8], 1.0, n_components=512, n_channels=3,
                         n_features_white=8 ** 2 * 3 - 1, dist='student', mean_mode='gc', sdev_mode='gc',
                         whiten_mode='pca',
                         name=None, load_name='FoEPrior', dir_name=None, load_tensor_names='image')
    elif mode == 'full512logistic':
        p = FoEFullPrior('pre_featmap:0', 1e-5, 'alexnet', [8, 8], 1.0, n_components=512, n_channels=3,
                         n_features_white=8 ** 2 * 3 - 1, dist='logistic', mean_mode='gc', sdev_mode='gc',
                         whiten_mode='pca', load_tensor_names='image')
    elif mode == 'dropout1024':
        p = FoEDropoutPrior('pre_featmap:0', 1e-5, 'alexnet', [8, 8], 1.0, n_components=1024, n_channels=3,
                            n_features_white=8 ** 2 * 3 - 1, dist='student', mean_mode='gc', sdev_mode='gc',
                            whiten_mode='pca', load_tensor_names='image',
                            activate_dropout=True, make_switch=False, dropout_prob=0.5)
    elif mode == 'dropout_nodrop_train512':
        p = FoEDropoutPrior('rgb_scaled:0', 1e-5, 'alexnet', [8, 8], 1.0, n_components=512, n_channels=3,
                            n_features_white=8 ** 2 * 3 - 1, dist='student', mean_mode='gc', sdev_mode='gc',
                            whiten_mode='pca', dir_name='student_dropout_prior_nodrop_training',
                            load_tensor_names='image',
                            activate_dropout=False, make_switch=False, dropout_prob=0.5)
    elif mode == 'dropout_nodrop_train256':
        p = FoEDropoutPrior('rgb_scaled:0', 1e-5, 'alexnet', [8, 8], 1.0, n_components=256, n_channels=3,
                            n_features_white=8 ** 2 * 3 - 1, dist='student', mean_mode='gc', sdev_mode='gc',
                            whiten_mode='pca', dir_name='student_dropout_prior_nodrop_training',
                            activate_dropout=False, make_switch=False, dropout_prob=0.5)
    elif mode == 'fullc1l6000':
        p = FoEFullPrior(tensor_names='conv1/lin:0', weighting=1e-5, classifier='alexnet',
                         filter_dims=[8, 8], input_scaling=1.0, n_components=6000, n_channels=96,
                         n_features_white=3000, dist='student', mean_mode='gc', sdev_mode='gc',
                         load_name='FoEPrior',
                         load_tensor_names='conv1/lin:0')
    elif mode == 'slimc5l5000':
        p = FoEFullPrior('pre_featmap/read:0', 1e-7, 'alexnet', [3, 3], 1.0, n_components=5000, n_channels=256,
                         n_features_white=3**2*256, dist='student', mean_mode='gc', sdev_mode='gc',
                         whiten_mode='pca',
                         name=None, load_name=None, dir_name=None, load_tensor_names='conv5/lin:0')
    else:
        p = FoEDropoutPrior('rgb_scaled:0', 1e-5, 'alexnet', [8, 8], 1.0, n_components=1024, n_channels=3,
                            n_features_white=8 ** 2 * 3 - 1, dist='student', mean_mode='gc', sdev_mode='gc',
                            whiten_mode='pca', dir_name='student_dropout_prior_nodrop_training',
                            activate_dropout=True, make_switch=False, dropout_prob=0.5)
    if custom_weighting is not None:
        p.weighting = custom_weighting
    return p
