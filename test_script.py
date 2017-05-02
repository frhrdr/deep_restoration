from layer_inversion import Parameters, LayerInversion
from filehandling_utils import selected_imgs

params = Parameters(classifier='alexnet', inv_input_name='conv1/relu:0', inv_target_name='bgr_normed:0',
                    inv_model='conv_deconv',
                    op1_height=11, op1_width=11, op1_strides=[1, 1, 1, 1],
                    op2_height=11, op2_width=11, op2_strides=[1, 4, 4, 1],
                    hidden_channels=96,
                    learning_rate=0.0003, batch_size=32, num_iterations=3000,
                    optimizer='adam',
                    data_path='./data/imagenet2012-validationset/',
                    train_images_file='train_48k_images.txt',
                    validation_images_file='validate_2k_images.txt',
                    log_path='./logs/layer_inversion/alexnet/l1_cd/run2/',
                    load_path='./logs/layer_inversion/alexnet/l1_cd/run2/ckpt-3000',
                    print_freq=100, log_freq=1000, test_freq=100, test_set_size=200)

li = LayerInversion(params)
# li.visualize(num_images=7, rec_type='bgr_normed', file_name='selected_diff')
li.visualize(num_images=7, rec_type='bgr_normed')
