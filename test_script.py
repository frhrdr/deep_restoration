from layer_inversion import LayerInversion, run_stacked_models
from parameter_utils import default_params, selected_images


specs1 = dict(op1_height=5, op1_width=5, op1_strides=[1, 2, 2, 1],
              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
              hidden_channels=64, target_shape=[224, 224, 64])
specs2 = dict(op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
              hidden_channels=64, target_shape=[224, 224, 64])
specs3 = dict(op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
              op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
              hidden_channels=64, target_shape=[224, 224, 3])

params = dict(classifier='vgg16', inv_input_name='pool1:0', inv_target_name='bgr_normed:0',
              inv_model_type='3_deconv_conv',
              inv_model_specs=[specs1, specs2, specs3],
              log_path='./logs/layer_inversion/vgg16/l123_dc_dc_dc/run1/',
              load_path='./logs/layer_inversion/vgg16/l123_dc_dc_dc/run1/ckpt-3000')
params.update(default_params())

li = LayerInversion(params)
li.visualize(num_images=7)

li = LayerInversion(selected_images(params))
li.visualize(file_name='selected_diff')
