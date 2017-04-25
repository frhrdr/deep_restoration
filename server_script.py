from alexnet_layer_1_inversion_model import AlexNetLayer1Inversion, Parameters
import tensorflow as tf

params = Parameters(conv_height=11, conv_width=11,
                    deconv_height=11, deconv_width=11, deconv_channels=96,
                    learning_rate=0.0001, batch_size=32, num_iterations=1000,
                    optimizer=tf.train.AdamOptimizer,
                    data_path='./data/imagenet2012-validationset/', images_file='val_images.txt',
                    log_path='./logs/alexnet_inversion_layer_2/run1/',
                    load_path='./logs/alexnet_inversion_layer_2/run1/ckpt-1000',
                    log_freq=1000, test_freq=-1)

AlexNetLayer1Inversion(params).train()
AlexNetLayer1Inversion(params).visualize()


params = Parameters(conv_height=5, conv_width=5,
                    deconv_height=5, deconv_width=5, deconv_channels=96,
                    learning_rate=0.0001, batch_size=32, num_iterations=1000,
                    optimizer=tf.train.AdamOptimizer,
                    data_path='./data/imagenet2012-validationset/', images_file='val_images.txt',
                    log_path='./logs/alexnet_inversion_layer_1/run1/',
                    load_path='./logs/alexnet_inversion_layer_1/run1/ckpt-1000',
                    log_freq=1000, test_freq=-1)

AlexNetLayer1Inversion(params).train()
AlexNetLayer1Inversion(params).visualize()


