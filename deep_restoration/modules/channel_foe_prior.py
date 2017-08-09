from modules.channel_ica_prior import ChannelICAPrior

class ChannelFoEPrior(ChannelICAPrior):

    def __init__(self, tensor_names, weighting, classifier, filter_dims, input_scaling, n_components, n_channels,
                 n_features_white,
                 trainable=False, name='ChannelFoEPrior', load_name='ChannelFoEPrior', dir_name='channel_foe_prior',
                 mean_mode='lc', sdev_mode='none'):
        super().__init__(tensor_names, weighting, classifier, filter_dims, input_scaling, n_components, n_channels,
                 n_features_white,
                 trainable=trainable, name=name, load_name=load_name, dir_name=dir_name,
                 mean_mode=mean_mode, sdev_mode=sdev_mode)
        raise NotImplementedError