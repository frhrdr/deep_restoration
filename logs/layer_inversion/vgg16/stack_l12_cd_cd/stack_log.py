from layer_inversion import Parameters, LayerInversion, run_stacked_models
from filehandling_utils import selected_imgs


params1 = Parameters(classifier='vgg16', inv_input_name='conv1_2/relu:0', inv_target_name='conv1_1/relu:0',
                     inv_model='conv_deconv',
                     op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
                     op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
                     hidden_channels=64,
                     learning_rate=0.0003, batch_size=32, num_iterations=3000,
                     optimizer='adam',
                     data_path='./data/imagenet2012-validationset/',
                     train_images_file='train_48k_images.txt',
                     validation_images_file='validate_2k_images.txt',
                     log_path='./logs/layer_inversion/vgg16/l2_cd/run1/',
                     load_path='./logs/layer_inversion/vgg16/l2_cd/run1/ckpt-3000',
                     print_freq=100, log_freq=1000, test_freq=100, test_set_size=200)

params2 = Parameters(classifier='vgg16', inv_input_name='conv1_1/relu:0', inv_target_name='bgr_normed:0',
                    inv_model='conv_deconv',
                    op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
                    op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
                    hidden_channels=64,
                    learning_rate=0.0001, batch_size=10, num_iterations=3000,
                    optimizer='adam',
                    data_path='./data/imagenet2012-validationset/',
                    train_images_file='train_48k_images.txt',
                    validation_images_file='validate_2k_images.txt',
                    log_path='./logs/layer_inversion/vgg16/l1_cd/run3/',
                    load_path='./logs/layer_inversion/vgg16/l1_cd/run3/ckpt-3000',
                    print_freq=100, log_freq=1000, test_freq=-1, test_set_size=2000)

run_stacked_models([selected_imgs(params1), selected_imgs(params2)], file_name='stacked_selected')
run_stacked_models([params1, params2])