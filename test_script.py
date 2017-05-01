from layer_inversion import Parameters, LayerInversion
import tensorflow as tf

params = Parameters(classifier='vgg16', inv_input_name='conv1_1/relu:0', inv_target_name='rgb_scaled:0',
                    inv_model='deconv_conv',
                    op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
                    op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
                    hidden_channels=64,
                    learning_rate=0.0001, batch_size=10, num_iterations=100,
                    optimizer=tf.train.AdamOptimizer,
                    data_path='./data/imagenet2012-validationset/',
                    train_images_file='train_48k_images.txt',
                    validation_images_file='validate_2k_images.txt',
                    log_path='./logs/layer_inversion/vgg16/l1_dc/run1/',
                    load_path='./logs/layer_inversion/vgg16/l1_dc/run1/ckpt-3000',
                    print_freq=10, log_freq=1000, test_freq=2, test_set_size=20)

li = LayerInversion(params)
li.train()