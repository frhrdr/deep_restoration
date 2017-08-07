from utils.preprocessing import make_channel_separate_patch_data, make_flattened_patch_data

# 5x5 channel mean: global channel, sdev: global channel
try:
    make_channel_separate_patch_data(num_patches=100000, ph=5, pw=5, classifier='alexnet', map_name='conv1/lin:0',
                                     n_channels=3,
                                     save_dir='../data/patches/alexnet/new/conv1_lin_5x5_24feats_channelwise_mean_gc_sdev_gc/',
                                     whiten_mode='pca', batch_size=100,
                                     mean_mode='global_channel', cov_mode='global_channel')
except:
    pass

# 5x5 channel mean: local full, sdev: global channel
try:
    make_channel_separate_patch_data(num_patches=100000, ph=5, pw=5, classifier='alexnet', map_name='conv1/lin:0',
                                     n_channels=96,
                                     save_dir='../data/patches/alexnet/new/conv1_lin_5x5_24feats_channelwise_mean_lf_sdev_gc/',
                                     whiten_mode='pca', batch_size=100,
                                     mean_mode='local_full', cov_mode='global_channel')
except:
    pass


# 8x8 channel mean: global channel, sdev: global channel
try:
    make_channel_separate_patch_data(num_patches=100000, ph=8, pw=8, classifier='alexnet', map_name='conv1/lin:0',
                                     n_channels=96,
                                     save_dir='../data/patches/alexnet/new/conv1_lin_8x8_63feats_channelwise_mean_gc_sdev_gc/',
                                     whiten_mode='pca', batch_size=100,
                                     mean_mode='global_channel', cov_mode='global_channel')
except:
    pass

# 8x8 channel mean: local full, sdev: global channel
try:
    make_channel_separate_patch_data(num_patches=100000, ph=8, pw=8, classifier='alexnet', map_name='conv1/lin:0',
                                     n_channels=96,
                                     save_dir='../data/patches/alexnet/new/conv1_lin_8x8_63feats_channelwise_mean_gc_sdev_gc/',
                                     whiten_mode='pca', batch_size=100,
                                     mean_mode='local_full', cov_mode='global_channel')
except:
    pass



# 5x5 flat mean: global channel, sdev: global channel 2400f
try:
    make_flattened_patch_data(num_patches=100000, ph=5, pw=5, classifier='alexnet', map_name='conv1/lin:0',
                              n_channels=96,
                              save_dir='../data/patches/alexnet/new/conv1_lin_5x5_2399feats_mean_gc_sdev_gc/',
                              n_feats_white=2399, whiten_mode='pca', batch_size=100,
                              mean_mode='global_channel', cov_mode='global_channel')
except:
    pass

# 5x5 flat mean: local full, sdev: global channel 2400f
try:
    make_flattened_patch_data(num_patches=100000, ph=5, pw=5, classifier='alexnet', map_name='conv1/lin:0',
                              n_channels=96,
                              save_dir='../data/patches/alexnet/new/conv1_lin_5x5_2399feats_mean_lf_sdev_gc/',
                              n_feats_white=2399, whiten_mode='pca', batch_size=100,
                              mean_mode='local_full', cov_mode='global_channel')
except:
    pass


# 5x5 flat mean: global channel, sdev: global channel 1200f
try:
    make_flattened_patch_data(num_patches=100000, ph=5, pw=5, classifier='alexnet', map_name='conv1/lin:0',
                              n_channels=96,
                              save_dir='../data/patches/alexnet/new/conv1_lin_1200feats_mean_gc_sdev_gc/',
                              n_feats_white=1200, whiten_mode='pca', batch_size=100,
                              mean_mode='global_channel', cov_mode='global_channel')
except:
    pass

# 5x5 flat mean: local full, sdev: global channel 1200f
try:
    make_flattened_patch_data(num_patches=100000, ph=5, pw=5, classifier='alexnet', map_name='conv1/lin:0',
                              n_channels=96,
                              save_dir='../data/patches/alexnet/new/conv1_lin_1200feats_mean_lf_sdev_gc/',
                              n_feats_white=1200, whiten_mode='pca', batch_size=100,
                              mean_mode='local_full', cov_mode='global_channel')
except:
    pass
