from copy import deepcopy

PARAMETER_KEYS = ['classifier',
                  'inv_model_type',
                  'inv_model_specs',
                  'learning_rate', 'batch_size', 'num_iterations',
                  'optimizer',
                  'data_path', 'train_images_file', 'validation_images_file',
                  'log_path', 'load_path',
                  'print_freq', 'log_freq', 'test_freq',
                  'test_set_size', 'channel_losses']

MODULE_KEYS = ['inv_input_name', 'inv_target_name', 'target_shape', 'rec_name', 'add_loss']


def check_params(params, check_specs=True):
    for key in PARAMETER_KEYS:
        assert key in params.keys(), 'missing parameter: ' + key
        assert params[key] is not None, 'parameter not set: ' + key

    if check_specs:
        got_rec = False
        for spec in params['inv_model_specs']:
            for key in MODULE_KEYS:
                assert key in spec.keys(), 'missing parameter: ' + key
                assert spec[key] is not None, 'parameter not set: ' + key
            if spec['rec_name'] == 'reconstruction':
                got_rec = True
        if not got_rec:
            print('WARNING: No Tensor designated as final reconstruction.')


def default_params():
    return dict(learning_rate=0.0003, batch_size=32, num_iterations=3000,
                optimizer='adam',
                data_path='./data/imagenet2012-validationset/',
                train_images_file='train_48k_images.txt',
                validation_images_file='validate_2k_images.txt',
                print_freq=100, log_freq=1000, test_freq=100, test_set_size=200,
                summary_freq=20,
                channel_losses=False)


def selected_images(params):
    p = deepcopy(params)
    p['data_path'] = './data/selected/'
    p['validation_images_file'] = 'images.txt'
    return p
