from utils.preprocessing import make_flattened_patch_data

make_flattened_patch_data(num_patches=100000, ph=5, pw=5, classifier='alexnet', map_name='conv2/lin:0',
                          n_channels=3,
                          save_dir='../data/patches/image/8x8_mean_gf_sdev_gf/',
                          n_feats_white=191, whiten_mode='pca', batch_size=100,
                          mean_mode='gf', sdev_mode='gf')