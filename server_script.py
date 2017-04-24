from vgg_layer_1_inversion_model import VggLayer1Inversion, Parameters


params = Parameters(conv_height=1, conv_width=1,
                    deconv_height=3, deconv_width=3, deconv_channels=64,
                    learning_rate=0.0001, batch_size=32, num_iterations=3000,
                    optimizer=tf.train.AdamOptimizer,
                    data_path='./data/imagenet2012-validationset/', images_file='val_images.txt',
                    log_path='./logs/vgg_inversion_layer_1/run3/',
                    load_path='./logs/vgg_inversion_layer_1/run3/ckpt-3000',
                    log_freq=1000, test_freq=-1)

VggLayer1Inversion(params).train()
VggLayer1Inversion(params).visualize()

params = Parameters(conv_height=3, conv_width=3,
                    deconv_height=3, deconv_width=3, deconv_channels=64,
                    learning_rate=0.0001, batch_size=32, num_iterations=3000,
                    optimizer=tf.train.AdamOptimizer,
                    data_path='./data/imagenet2012-validationset/', images_file='val_images.txt',
                    log_path='./logs/vgg_inversion_layer_1/run4/',
                    load_path='./logs/vgg_inversion_layer_1/run4/ckpt-3000',
                    log_freq=1000, test_freq=-1)

VggLayer1Inversion(params).train()
VggLayer1Inversion(params).visualize()

params = Parameters(conv_height=5, conv_width=5,
                    deconv_height=5, deconv_width=5, deconv_channels=64,
                    learning_rate=0.0001, batch_size=32, num_iterations=3000,
                    optimizer=tf.train.AdamOptimizer,
                    data_path='./data/imagenet2012-validationset/', images_file='val_images.txt',
                    log_path='./logs/vgg_inversion_layer_1/run5/',
                    load_path='./logs/vgg_inversion_layer_1/run5/ckpt-3000',
                    log_freq=1000, test_freq=-1)

VggLayer1Inversion(params).train()
VggLayer1Inversion(params).visualize()