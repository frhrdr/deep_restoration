from alexnet_layer_1_inversion_model import AlexNetLayer1Inversion, Parameters
import tensorflow as tf

# params = Parameters(conv_height=11, conv_width=11,
#                     deconv_height=11, deconv_width=11, deconv_channels=96,
#                     learning_rate=0.0001, batch_size=10, num_iterations=1000,
#                     optimizer=tf.train.AdamOptimizer,
#                     data_path='./data/imagenet2012-validationset/', images_file='images.txt',
#                     log_path='./logs/alexnet_inversion_layer_1/run2/',
#                     load_path='./logs/alexnet_inversion_layer_1/run2/ckpt-1000',
#                     log_freq=1000, test_freq=-1)
#
# AlexNetLayer1Inversion(params).visualize(img_idx=1)

params = Parameters(conv_height=5, conv_width=5,
                    deconv_height=5, deconv_width=5, deconv_channels=96,
                    learning_rate=0.0001, batch_size=10, num_iterations=1000,
                    optimizer=tf.train.AdamOptimizer,
                    data_path='./data/imagenet2012-validationset/',
                    train_images='train_48k_images.txt',
                    log_path='./logs/alexnet_inversion_layer_1/run1/',

                    load_path='./logs/alexnet_inversion_layer_1/run1/ckpt-1000',
                    log_freq=1000, test_freq=-1)

for idx in range(5):
    AlexNetLayer1Inversion(params).visualize(img_idx=idx)


