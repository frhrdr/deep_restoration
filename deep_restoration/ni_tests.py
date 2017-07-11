from net_inversion2 import NetInversion
from modules.ica_prior import ICAPrior
from modules.loss_modules import MSELoss
from modules.split_module import SplitModule
from utils.parameter_utils import default_params

split = SplitModule(name_to_split='conv3/relu:0', img_slice_name='img_rep', rec_slice_name='rec_rep')
mse = MSELoss(target='img_rep', reconstruction='rec_rep', weighting=1.0)
ica = ICAPrior(tensor_names='reconstruction/read:0',
               weighting=1.0e-4, name='ICAPrior',
               load_path='./logs/priors/ica_prior/8by8_512_color/ckpt-10000',
               trainable=False, filter_dims=[8, 8], input_scaling=1.0, n_components=512)

modules = [split, mse, ica]

params = dict(classifier='alexnet',
              modules=modules,
              log_path='./logs/mahendran_vedaldi/2016/',
              load_path='')
params.update(default_params())
params['learning_rate'] = 0.04
params['optimizer'] = 'momentum'

ni = NetInversion(params)

ni.train_pre_image('../data/selected/images_resized/val13_monkey.bmp')

