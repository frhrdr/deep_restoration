from utils.temp_utils import make_feat_map_mats, make_reduced_feat_map_mats

make_feat_map_mats(num_patches=100000, map_name='conv2/relu:0', classifier='alexnet', ph=5, pw=5,
                   save_dir='../data/patches/alexnet/conv2_relu_5x5/', whiten_mode='pca', batch_size=50)

make_reduced_feat_map_mats(100000, load_dir='../data/patches/alexnet/conv2_relu_5x5/',
                           n_features=6400, n_to_keep=3200,
                           save_dir='../data/patches/alexnet/conv2_relu_5x5_3200feats/')
