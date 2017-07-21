from utils.temp_utils import make_feat_map_mats, make_reduced_feat_map_mats

# make_feat_map_mats(num_patches=100000, map_name='conv3/relu:0', classifier='alexnet', ph=5, pw=5,
#                    save_dir='../data/patches/alexnet/conv3_relu_5x5/', whiten_mode='pca', batch_size=10)

make_reduced_feat_map_mats(100000, load_dir='../data/patches/alexnet/conv3_relu_5x5/',
                           n_features=9600, n_to_keep=1000,
                           save_dir='../data/patches/alexnet/conv3_relu_5x5_1000feats/')