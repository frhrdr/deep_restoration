from utils.preprocessing import make_flattened_patch_data
from utils.temp_utils import analyze_eigenvals

make_flattened_patch_data(num_patches=100000, ph=5, pw=5, classifier='alexnet', map_name='lrn1:0',
                          n_channels=96,
                          save_dir='../data/patches/alexnet/lrn1_5x5_1000feats_mean_lc_sdev_gc/',
                          n_feats_white=1000, whiten_mode='pca', batch_size=100,
                          mean_mode='lc', sdev_mode='gc')


analyze_eigenvals('../data/patches/alexnet/lrn1_5x5_1000feats_mean_lc_sdev_gc/cov.npy', n_to_drop=1000)