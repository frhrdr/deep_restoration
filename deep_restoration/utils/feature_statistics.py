import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('qt5agg', warn=False, force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from utils.filehandling import get_feature_files
from sklearn.decomposition import PCA, FastICA


layer_list = ["conv1_1", "conv1_2", "pool1", "conv2_1", "conv2_2", "pool2",
              "conv3_1", "conv3_2", "conv3_3", "pool3", "conv4_1", "conv4_2", "conv4_3", "pool4",
              "conv5_1", "conv5_2", "conv5_3", "pool5", "fc6", "relu6",
              "fc7", "relu7", "fc8", "prob"]


def gini_index(feature_map):
    feature_vec = np.sort(feature_map.flatten(), axis=0)
    norm = np.sum(np.abs(feature_vec))
    acc = 0.0
    n = feature_vec.shape[0]
    for k, feat in enumerate(feature_vec):
        acc += feat * (n - k + 0.5)
    acc *= 2.0 / (norm * n)
    gini = 1.0 - acc
    return gini


def gini_index_hist(layer_name, subset_file='subset_cutoff_200_images.txt'):
    files = get_feature_files(layer_name, subset_file)
    ginis = []
    for idx, file in enumerate(files):
        if idx % 10 == 0:
            print('processing file ' + str(idx) + ' / ' + str(len(files)))
        mat = np.load(file)
        ginis.append(gini_index(mat))
    plt.hist(ginis, 50, normed=1, facecolor='green', alpha=0.75)
    plt.savefig('./plots/gini_hist_' + layer_name + '.png', format='png')


def gram_matrix(feature_maps):
    assert len(feature_maps.shape) == 3
    num_maps = feature_maps.shape[-1]
    feature_maps = np.reshape(np.transpose(feature_maps, (2, 0, 1)), (num_maps, -1))
    feature_maps -= feature_maps.mean()
    gram = np.dot(feature_maps, feature_maps.T)
    return gram


def avg_gram_matrix(layer_name, subset_file='subset_cutoff_200_images.txt'):
    files = get_feature_files(layer_name, subset_file)
    avg_gram = None
    for idx, file in enumerate(files):
        if idx % 10 == 0:
            print('processing file ' + str(idx) + ' / ' + str(len(files)))
        mat = np.load(file)
        if avg_gram is None:
            avg_gram = gram_matrix(mat)
        else:
            avg_gram += gram_matrix(mat)
    avg_gram /= len(files)
    plt.matshow(avg_gram, interpolation='none')
    plt.savefig('./plots/avg_gram_' + layer_name + '.png', format='png')


def gatys_gram_loss(feat_maps_a, feat_maps_b):
    n = feat_maps_a.shape[2]
    m = feat_maps_a.shape[0] * feat_maps_a.shape[1]
    norm = 4 * n * n * m * m
    gram_a = gram_matrix(feat_maps_a)
    gram_b = gram_matrix(feat_maps_b)
    diff = gram_a - gram_b
    sum_diff = np.sum(np.multiply(diff, diff))
    loss = sum_diff / norm
    return loss


def covariance_matrix(feature_map):
    assert len(feature_map.shape) == 3
    num_maps = feature_map.shape[-1]
    feature_map = np.reshape(np.transpose(feature_map, (2, 0, 1)), (num_maps, -1))
    # feature_map -= feature_map.mean()
    # cov = np.dot(feature_map.T, feature_map)
    cov = np.cov(feature_map.T)
    return cov


def avg_covariance_matrix(layer_name, subset_file='subset_cutoff_200_images.txt'):
    files = get_feature_files(layer_name, subset_file)
    avg_cov = None
    for idx, file in enumerate(files):
        if idx % 10 == 0:
            print('processing file ' + str(idx) + ' / ' + str(len(files)))
        mat = np.load(file)
        if avg_cov is None:
            avg_cov = covariance_matrix(mat)
        else:
            avg_cov += covariance_matrix(mat)
    avg_cov /= len(files)
    plt.matshow(avg_cov, interpolation='none')
    plt.savefig('./plots/avg_cov_' + layer_name + '.png', format='png', dpi=1500)


def inverse_covariance_matrix(feature_map):
    cov = covariance_matrix(feature_map)
    return np.linalg.pinv(cov)


def inverse_avg_covariance_matrix(layer_name, subset_file='subset_cutoff_200_images.txt', log=False):
    files = get_feature_files(layer_name, subset_file)
    avg_cov = None
    for idx, file in enumerate(files):
        if idx % 10 == 0:
            print('processing file ' + str(idx) + ' / ' + str(len(files)))
        mat = np.load(file)
        if avg_cov is None:
            avg_cov = covariance_matrix(mat)
        else:
            avg_cov += covariance_matrix(mat)
    avg_cov /= len(files)
    inv_cov = np.linalg.pinv(avg_cov)
    inv_cov[inv_cov == np.inf] = inv_cov[inv_cov != np.inf].max()
    if log:
        inv_cov[inv_cov == -np.inf] = inv_cov[inv_cov != -np.inf].min()
        inv_cov += inv_cov.min() + 0.0000000001
        norm = LogNorm(vmin=inv_cov.min(), vmax=inv_cov.max())
        plt.matshow(inv_cov, norm=norm, interpolation='none')
        plt.savefig('./plots/inv_avg_cov_' + layer_name + '_log.png', format='png', dpi=1500)
    else:
        inv_cov[inv_cov == -np.inf] = inv_cov[inv_cov != -np.inf].min()
        plt.matshow(inv_cov, interpolation='none')
        plt.savefig('./plots/inv_avg_cov_' + layer_name + '_lin.png', format='png', dpi=1500)


def feat_map_vis(feature_map, max_n, highest_act):
    assert len(feature_map.shape) == 3
    num_maps = feature_map.shape[-1]
    feature_map = np.transpose(feature_map, (2, 0, 1))
    if highest_act:
        mat = np.reshape(feature_map, (num_maps, -1))
        means = np.mean(np.abs(mat), axis=1)
        sort_ids = np.argsort(means)[::-1]
        mat = feature_map[sort_ids, :, :]
    else:
        mat = feature_map

    if mat.shape[0] > max_n:
        mat = mat[:max_n, :, :]

    n = mat.shape[0]
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n // cols))

    fig, ax_list = plt.subplots(ncols=cols, nrows=rows)
    ax_list = ax_list.flatten()
    for idx, ax in enumerate(ax_list):
        if idx >= n:
            ax.axis('off')
        else:
            ax.matshow(mat[idx, :, :], interpolation='none')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()


def visualize_feature_map(layer_name, subset_file='subset_cutoff_200_images.txt',
                          image_index=0, max_n=25, highest_act=True):
    file = get_feature_files(layer_name, subset_file)[image_index]
    feat_map = np.load(file)
    feat_map_vis(feat_map, max_n, highest_act)


def feat_map_pca(layer_name, subset_file='subset_cutoff_200_images.txt', map_index=0, n_plots=25):
    files = get_feature_files(layer_name, subset_file)
    maps = []
    for idx, file in enumerate(files):
        mat = np.load(file)
        map_shape = mat[:, :, map_index].shape
        maps.append(mat[:, :, map_index].flatten())
    maps = np.stack(maps, axis=0)
    pca = PCA()
    pca.fit(maps)

    cols = int(np.ceil(np.sqrt(n_plots)))
    rows = int(np.ceil(n_plots // cols))
    fig, ax_list = plt.subplots(ncols=cols, nrows=rows)
    ax_list = ax_list.flatten()
    for idx, ax in enumerate(ax_list):
        if idx >= n_plots:
            ax.axis('off')
        else:
            ax.matshow(np.reshape(pca.components_[idx, :], map_shape), interpolation='none')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.savefig('./plots/pca_' + layer_name + '_map_' + str(map_index) + '.png', format='png', dpi=1500)
    plt.close()


def feat_map_ica(layer_name, subset_file='subset_cutoff_200_images.txt', map_index=0, n_plots=25):
    files = get_feature_files(layer_name, subset_file)
    maps = []
    for idx, file in enumerate(files):
        mat = np.load(file)
        map_shape = mat[:, :, map_index].shape
        maps.append(mat[:, :, map_index].flatten())
    maps = np.stack(maps, axis=0)
    ica = FastICA()
    ica.fit(maps)

    cols = int(np.ceil(np.sqrt(n_plots)))
    rows = int(np.ceil(n_plots // cols))
    fig, ax_list = plt.subplots(ncols=cols, nrows=rows)
    ax_list = ax_list.flatten()
    for idx, ax in enumerate(ax_list):
        if idx >= n_plots:
            ax.axis('off')
        else:
            ax.matshow(np.reshape(ica.components_[idx, :], map_shape), interpolation='none')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.savefig('./plots/ica_' + layer_name + '_map_' + str(map_index) + '.png', format='png', dpi=1500)
    plt.close()


def inv_avg_gram_matrix(layer_name, subset_file='subset_cutoff_200_images.txt'):
    files = get_feature_files(layer_name, subset_file)
    avg_gram = None
    for idx, file in enumerate(files):
        if idx % 10 == 0:
            print('processing file ' + str(idx) + ' / ' + str(len(files)))
        mat = np.load(file)
        if avg_gram is None:
            avg_gram = gram_matrix(mat)
        else:
            avg_gram += gram_matrix(mat)
    avg_gram /= len(files)
    inv_gram = np.linalg.pinv(avg_gram)
    plt.matshow(inv_gram, interpolation='none')
    plt.savefig('./plots/inv_avg_gram_' + layer_name + '.png', format='png', dpi=1500)


def gram_tensor(feature_map):
    map_shape = [k.value for k in feature_map.get_shape()]
    assert len(map_shape) == 3
    num_maps = map_shape[-1]
    feature_map = tf.reshape(tf.transpose(feature_map, perm=(2, 0, 1)), shape=(num_maps, -1))
    feature_map -= tf.reduce_mean(feature_map)
    gram = tf.matmul(feature_map, feature_map, transpose_b=True)
    return gram