PARAMETER_KEYS = ['classifier', 'inv_input_name', 'inv_target_name',
                  'inv_model_type',
                  'inv_model_specs',
                  'learning_rate', 'batch_size', 'num_iterations',
                  'optimizer',
                  'data_path', 'train_images_file', 'validation_images_file',
                  'log_path', 'load_path',
                  'print_freq', 'log_freq', 'test_freq',
                  'test_set_size', 'channel_losses']


def check_params(params):
    for key in PARAMETER_KEYS:
        assert key in params.keys(), 'missing parameter: ' + key
        assert params[key] is not None, 'parameter not set: ' + key


def default_params():
    return dict(learning_rate=0.0003, batch_size=32, num_iterations=3000,
                optimizer='adam',
                data_path='./data/imagenet2012-validationset/',
                train_images_file='train_48k_images.txt',
                validation_images_file='validate_2k_images.txt',
                print_freq=100, log_freq=1000, test_freq=100, test_set_size=200,
                channel_losses=False)


def selected_images(params):
    params['data_path'] = './data/selected/'
    params['validation_images_file'] = 'images.txt'
    return params
