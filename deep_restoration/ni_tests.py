from net_inversion import NetInversion
from modules.ica_prior import ICAPrior
from modules.loss_modules import NormedMSELoss
from modules.split_module import SplitModule
from utils.parameter_utils import mv_default_params

split = SplitModule(name_to_split='conv4/relu:0', img_slice_name='img_rep', rec_slice_name='rec_rep')
mse = NormedMSELoss(target='img_rep:0', reconstruction='rec_rep:0', weighting=1.0)
ica = ICAPrior(tensor_names='reconstruction/read:0',
               weighting=1.0e-3, name='ICAPrior',
               load_path='../logs/priors/ica_prior/8by8_512_color/ckpt-10000',
               trainable=False, filter_dims=[8, 8], input_scaling=1.0, n_components=512)

modules = [split, mse, ica]

params = dict(classifier='alexnet',
              modules=modules,
              log_path='../logs/mahendran_vedaldi/2016/',
              load_path='')
params.update(mv_default_params())
params['num_iterations'] = 10000

ni = NetInversion(params)

ni.train_pre_image('../data/selected/images_resized/val13_monkey.bmp', jitter_t=0, optim_name='adam',
                   lr_lower_points=((0, 0.01), (1000, 0.004), (1200, 0.001), (3000, 0.0004), (6000, 0.0001), (9000, 0.00004)))
