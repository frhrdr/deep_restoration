from layer_inversion import LayerInversion, run_stacked_models
from parameter_utils import default_params

specs = dict(op1_height=5, op1_width=5, op1_strides=[1, 1, 1, 1],
             op2_height=5, op2_width=5, op2_strides=[1, 1, 1, 1],
             hidden_channels=64)
params = dict(classifier='vgg16', inv_input_name='conv1_1/relu:0', inv_target_name='bgr_normed:0',
              inv_model_type='conv_deconv',
              inv_model_specs=specs,
              log_path='./logs/layer_inversion/vgg16/l1_cd/run3/',
              load_path='./logs/layer_inversion/vgg16/l1_cd/run3/ckpt-3000')
params.update(default_params())
print(params)

li = LayerInversion(params)
li.visualize(num_images=5, rec_type='bgr_normed', file_name='img_vs_rec2', add_diffs=True)