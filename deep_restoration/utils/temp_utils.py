import numpy as np
from utils.filehandling_utils import load_image
import os
import skimage
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
from tf_vgg.vgg16 import Vgg16
from tf_alexnet.alexnet import AlexNet
import tensorflow as tf


def flattening_filter(dims):
    assert len(dims) == 3
    f = np.zeros((dims[0], dims[1], dims[2], dims[0] * dims[1] * dims[2]))
    for idx in range(f.shape[3]):
        x = (idx // (dims[1] * dims[2])) % dims[0]
        y = (idx // dims[2]) % dims[1]
        z = idx % dims[2]

        f[x, y, z, idx] = 1
    return f


def pca_whiten_as_ica(data):
    n_samples, n_features = data.shape
    data = data.T
    data -= data.mean(axis=-1)[:, np.newaxis]
    u, d, _ = np.linalg.svd(data, full_matrices=False)
    K = (u / d).T[:n_features]
    data = np.dot(K, data)
    data *= np.sqrt(n_samples)
    return data.T, K


def pca_whiten_as_pca(data):
    n_samples, n_features = data.shape
    u, s, v = np.linalg.svd(data, full_matrices=False)
    max_abs_cols = np.argmax(np.abs(u), axis=0)
    signs = np.sign(u[max_abs_cols, range(u.shape[1])])
    u *= signs
    u *= np.sqrt(n_samples)
    v *= signs[:, np.newaxis]
    rerotate = v / s[:, np.newaxis] * np.sqrt(n_samples)
    return u, rerotate


def pca_whiten_mats(cov, n_to_drop=1):
    e_vals, e_vecs = np.linalg.eigh(cov)
    assert all(a <= b for a, b in zip(e_vals[:-1], e_vals[1:]))  # make sure vals are sorted ascending (they should be)
    # print(e_vals.sum())
    # print(e_vals)
    full_eigv = sum(e_vals)
    keep_eigv = sum(e_vals[n_to_drop:])

    print('kept eigv fraction: ', keep_eigv / full_eigv)
    print(e_vals)
    e_vals = e_vals[n_to_drop:]  # dismiss first eigenvalue due to mean subtraction.
    e_vecs = e_vecs[:, n_to_drop:]
    sqrt_vals = np.sqrt(np.maximum(e_vals, 0))
    whiten = np.diag(1. / sqrt_vals) @ e_vecs.T
    un_whiten = e_vecs @ np.diag(sqrt_vals)
    return whiten, un_whiten.T


def pca_whiten(data):
    cov = np.dot(data.T, data) / (data.shape[0] - 1)
    whiten, unwhiten = pca_whiten_mats(cov)
    data = np.dot(data, whiten.T)
    return data, unwhiten


def zca_whiten_mats(cov):
    U, S, _ = np.linalg.svd(cov)
    s = np.sqrt(S.clip(1.e-6))
    s_inv = np.diag(1. / s)
    s = np.diag(s)
    whiten = np.dot(np.dot(U, s_inv), U.T)
    un_whiten = np.dot(np.dot(U, s), U.T)
    return whiten, un_whiten


def zca_whiten(data):
    cov = np.dot(data.T, data) / (data.shape[0] - 1)
    whiten, unwhiten = zca_whiten_mats(cov)
    data = np.dot(data, whiten.T)
    return data, unwhiten


def make_data_mat_from_patches(data_dir='./data/patches_gray/8x8/', whiten_mode='pca'):
    data_set_size = len([name for name in os.listdir(data_dir) if name.startswith('patch')])
    mm = np.memmap(data_dir + '/data_mat_' + whiten_mode + '.npy', dtype=np.float32, mode='w+',
                   shape=(data_set_size, 63))
    if whiten_mode == 'pca' or whiten_mode == 'zca':
        cov_acc = np.load(data_dir + '/cov.npy')

        if whiten_mode == 'pca':
            whiten, un_whiten = pca_whiten_mats(cov_acc)
        else:
            whiten, un_whiten = zca_whiten_mats(cov_acc)

    for idx in range(data_set_size):

        image = load_image(data_dir + 'patch_{}.bmp'.format(idx), resize=False)
        image = image.flatten().astype(np.float32)
        image /= 255.0
        image -= image.mean()
        image = whiten @ image
        mm[idx, :] = image


def make_cov_acc(data_dir='./data/patches_gray/8x8/'):
    data_set_size = len([name for name in os.listdir(data_dir) if name.startswith('patch')])
    cov_acc = 0
    for idx in range(data_set_size):
        image = load_image(data_dir + 'patch_{}.bmp'.format(idx), resize=False).flatten()
        image = image.astype(np.float32) / 255.0
        image -= image.mean()
        cov_acc = cov_acc + np.outer(image, image)

        if idx % (data_set_size / 10) == 0:
            print('cov progress: ', idx, ' / ', data_set_size)

    cov_acc = cov_acc.astype(np.float32) / (data_set_size - 1)
    np.save(data_dir + '/cov.npy', cov_acc)


def make_data_mats(num_patches, ph=8, pw=8, color=False, save_dir='./data/patches_gray/new8by8/', whiten_mode='pca'):
    img_hw = 224
    max_h = img_hw - ph
    max_w = img_hw - pw
    data_path = './data/imagenet2012-validationset/'
    img_file = 'train_48k_images.txt'
    n_features = ph * pw
    if color:
        n_features *= 3
    raw_mat = np.memmap(save_dir + 'data_mat_raw.npy', dtype=np.float32, mode='w+',
                        shape=(num_patches, n_features))

    with open(data_path + img_file) as f:
        image_files = [k.rstrip() for k in f.readlines()]

    image_paths = [data_path + 'images_resized/' +
                   k[:-len('JPEG')] + 'bmp' for k in image_files]

    cov_acc = 0
    for idx in range(num_patches):
        img_path = image_paths[idx % len(image_paths)]
        h = np.random.randint(0, max_h)
        w = np.random.randint(0, max_w)
        image = load_image(img_path, resize=False)
        image = image[h:h + ph, w:w + pw, :]
        if not color:
            image = skimage.rgb2gray(image)
        image = image.flatten().astype(np.float32)
        image /= 255.0  # map to range [0,1]
        image -= image.mean()  # subtract image mean

        raw_mat[idx, :] = image
        cov_acc = cov_acc + np.outer(image, image)

    cov = cov_acc / (num_patches - 1)
    np.save(save_dir + 'cov.npy', cov)

    raw_mat.flush()
    del raw_mat

    print('raw mat and cov done')

    raw_mat = np.memmap(save_dir + '/data_mat_raw.npy', dtype=np.float32, mode='r',
                        shape=(num_patches, n_features))

    if whiten_mode == 'pca':
        whiten, un_whiten = pca_whiten_mats(cov)
    elif whiten_mode == 'zca':
        whiten, un_whiten = zca_whiten_mats(cov)
    else:
        raise NotImplementedError

    np.save(save_dir + 'whiten_' + whiten_mode + '.npy', whiten)
    np.save(save_dir + 'unwhiten_' + whiten_mode + '.npy', un_whiten)

    data_mat = np.memmap(save_dir + 'data_mat_' + whiten_mode + '_whitened.npy', dtype=np.float32, mode='w+',
                         shape=(num_patches, whiten.shape[0]))

    for idx in range(num_patches):
        image = raw_mat[idx, :]
        image = whiten @ image
        data_mat[idx, :] = image


def make_feat_map_mats(num_patches, map_name, classifier='alexnet', ph=8, pw=8,
                       save_dir='./data/patches/', whiten_mode='pca', batch_size=10):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    assert num_patches % batch_size == 0

    if classifier.lower() == 'vgg16':
        classifier = Vgg16()
    elif classifier.lower() == 'alexnet':
        classifier = AlexNet()
    else:
        raise NotImplementedError

    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:

            img_pl = tf.placeholder(dtype=tf.float32, shape=[batch_size, 224, 224, 3])
            classifier.build(img_pl, rescale=1.0)
            feat_map = graph.get_tensor_by_name(map_name)
            map_dims = [d.value for d in feat_map.get_shape()]
            n_features = ph * pw * map_dims[3]
            print(n_features)
            data_path = '../data/imagenet2012-validationset/'
            img_file = 'train_48k_images.txt'

            raw_mat = np.memmap(save_dir + 'data_mat_raw.npy', dtype=np.float32, mode='w+',
                                shape=(num_patches, n_features))
            max_h = map_dims[1] - ph
            max_w = map_dims[2] - pw

            with open(data_path + img_file) as f:
                image_files = [k.rstrip() for k in f.readlines()]

            image_paths = [data_path + 'images_resized/' + k[:-len('JPEG')] + 'bmp' for k in image_files]
            img_mat = np.zeros(shape=[batch_size, 224, 224, 3])

            cov_acc = 0

            for count in range(num_patches // batch_size):

                for idx in range(batch_size):
                    img_path = image_paths[idx + (count * batch_size) % len(image_paths)]
                    img_mat[idx, :, :, :] = load_image(img_path, resize=False)

                map_mat = sess.run(feat_map, feed_dict={img_pl: img_mat})
                for idx in range(batch_size):
                    h = np.random.randint(0, max_h)
                    w = np.random.randint(0, max_w)

                    map_patch = map_mat[idx, h:h + ph, w:w + pw, :]
                    map_patch = map_patch.flatten().astype(np.float32)
                    map_patch -= map_patch.mean()  # subtract image mean

                    raw_mat[idx + (count * batch_size), :] = map_patch
                    cov_acc = cov_acc + np.outer(map_patch, map_patch)

                    if idx + (count * batch_size) % (num_patches // 100) == 0:
                        print(100 * (idx + count * batch_size) / num_patches, '% cov accumulation done')
            cov = cov_acc / (num_patches - 1)
            np.save(save_dir + 'cov.npy', cov)

            raw_mat.flush()
            del raw_mat

    print('raw mat and cov done')

    raw_mat = np.memmap(save_dir + '/data_mat_raw.npy', dtype=np.float32, mode='r',
                        shape=(num_patches, n_features))

    if whiten_mode == 'pca':
        whiten, un_whiten = pca_whiten_mats(cov)
    elif whiten_mode == 'zca':
        whiten, un_whiten = zca_whiten_mats(cov)
    else:
        raise NotImplementedError

    np.save(save_dir + 'whiten_' + whiten_mode + '.npy', whiten)
    np.save(save_dir + 'unwhiten_' + whiten_mode + '.npy', un_whiten)

    data_mat = np.memmap(save_dir + 'data_mat_' + whiten_mode + '_whitened.npy', dtype=np.float32, mode='w+',
                         shape=(num_patches, whiten.shape[0]))

    for idx in range(num_patches):
        image = raw_mat[idx, :]
        image = whiten @ image
        data_mat[idx, :] = image

    data_mat.flush()
    del data_mat


def make_reduced_feat_map_mats(num_patches, load_dir, n_features, n_to_keep,
                               save_dir='./data/patches/', whiten_mode='pca'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    raw_mat = np.memmap(load_dir + '/data_mat_raw.npy', dtype=np.float32, mode='r',
                        shape=(num_patches, n_features))
    cov = np.load(load_dir + 'cov.npy')

    print('raw mat and cov loaded')

    if whiten_mode == 'pca':
        whiten, un_whiten = pca_whiten_mats(cov, n_to_drop=n_features - n_to_keep)
    elif whiten_mode == 'zca':
        raise NotImplementedError
    else:
        raise NotImplementedError

    np.save(save_dir + 'whiten_' + whiten_mode + '.npy', whiten)
    np.save(save_dir + 'unwhiten_' + whiten_mode + '.npy', un_whiten)

    data_mat = np.memmap(save_dir + 'data_mat_' + whiten_mode + '_whitened.npy', dtype=np.float32, mode='w+',
                         shape=(num_patches, whiten.shape[0]))

    for idx in range(num_patches):
        image = raw_mat[idx, :]
        image = whiten @ image
        data_mat[idx, :] = image

    data_mat.flush()
    del data_mat


def make_channel_separate_feat_map_mats(num_patches, ph, pw, classifier, map_name, n_channels,
                                        save_dir, whiten_mode='pca', batch_size=10):
    """
    creates whitening, covariance, raw and whitened feature matrices for separate channels.
    They are saved as 3d matrices where the first dimension is the channel index

    :param num_patches: number of patches in the raw data mat
    :param load_dir: take raw data mat and accumulated cov from here
    :param n_features: number of patch features summed over all channels
    :param n_channels: number of channels
    :param save_dir: place to save
    :param whiten_mode: 'pca' or 'zca'. only pca whitening is supported atm
    :return: None. objects are saved
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    assert num_patches % batch_size == 0

    if classifier.lower() == 'vgg16':
        classifier = Vgg16()
    elif classifier.lower() == 'alexnet':
        classifier = AlexNet()
    else:
        raise NotImplementedError

    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:

            img_pl = tf.placeholder(dtype=tf.float32, shape=[batch_size, 224, 224, 3], name='img_pl')
            classifier.build(img_pl, rescale=1.0)
            feat_map = graph.get_tensor_by_name(map_name)
            map_dims = [d.value for d in feat_map.get_shape()]
            n_feats_per_channel = ph * pw
            n_features = n_feats_per_channel * map_dims[3]

            data_path = '../data/imagenet2012-validationset/'
            img_file = 'train_48k_images.txt'

            raw_mat = np.memmap(save_dir + 'data_mat_raw_channel.npy', dtype=np.float32, mode='w+',
                                shape=(num_patches, n_channels, n_feats_per_channel))

            max_h = map_dims[1] - ph
            max_w = map_dims[2] - pw

            with open(data_path + img_file) as f:
                image_files = [k.rstrip() for k in f.readlines()]

            image_paths = [data_path + 'images_resized/' + k[:-len('JPEG')] + 'bmp' for k in image_files]
            img_mat = np.zeros(shape=[batch_size, 224, 224, 3])

            channel_covs = np.zeros(shape=[n_channels, n_feats_per_channel, n_feats_per_channel])

            for count in range(num_patches // batch_size):

                for idx in range(batch_size):
                    img_path = image_paths[idx + (count * batch_size) % len(image_paths)]
                    img_mat[idx, :, :, :] = load_image(img_path, resize=False)

                if count == 0:
                    print('Verifying scale - this should be around 255: ', np.max(img_mat))
                map_mat = sess.run(feat_map, feed_dict={img_pl: img_mat})

                for idx in range(batch_size):
                    h = np.random.randint(0, max_h)
                    w = np.random.randint(0, max_w)
                    map_patch = np.rollaxis(map_mat[idx, h:h + ph, w:w + pw, :], axis=2)

                    map_patch = map_patch.reshape([n_channels, -1]).astype(np.float32)
                    map_patch = (map_patch.T - map_patch.mean(axis=1)).T

                    raw_mat[idx + (count * batch_size), :, :] = map_patch
                    acc = np.matmul(map_patch.reshape([n_channels, -1, 1]), map_patch.reshape([n_channels, 1, -1]))
                    channel_covs = channel_covs + acc

                    if idx + (count * batch_size) % (num_patches // 100) == 0:
                        print(100 * (idx + count * batch_size) / num_patches, '% cov accumulation done')

            channel_covs = channel_covs / (num_patches - 1)
            np.save(save_dir + 'channel_covs.npy', channel_covs)

            raw_mat.flush()
            del raw_mat

    print('raw mat and cov done')

    channel_whiten = np.zeros(shape=[n_channels, n_feats_per_channel - 1, n_feats_per_channel])
    channel_unwhiten = np.zeros(shape=[n_channels, n_feats_per_channel - 1, n_feats_per_channel])

    for idx in range(n_channels):

        if whiten_mode == 'pca':
            whiten, unwhiten = pca_whiten_mats(channel_covs[idx, :, :], n_to_drop=1)
        elif whiten_mode == 'zca':
            raise NotImplementedError
        else:
            raise NotImplementedError

        channel_whiten[idx, :, :] = whiten
        channel_unwhiten[idx, :, :] = unwhiten

    np.save(save_dir + 'channel_whiten_' + whiten_mode + '.npy', channel_whiten)
    np.save(save_dir + 'channel_unwhiten_' + whiten_mode + '.npy', channel_unwhiten)
    print('whitening mats saved')

    data_mat = np.memmap(save_dir + 'data_mat_' + whiten_mode + '_channel_whitened.npy', dtype=np.float32, mode='w+',
                         shape=(num_patches, n_channels, channel_whiten.shape[1]))

    raw_mat = np.memmap(save_dir + 'data_mat_raw_channel.npy', dtype=np.float32, mode='r',
                        shape=(num_patches, n_features))

    for idx in range(num_patches):
        image = raw_mat[idx, :].reshape([n_feats_per_channel, n_channels]).T
        image = np.expand_dims(image, axis=2)
        data_mat[idx, :] = np.squeeze(channel_whiten @ image)

    data_mat.flush()
    del data_mat
    del raw_mat


def patch_batch_gen(batch_size, data_dir='./data/patches_gray/new8by8/', whiten_mode='pca',
                    data_shape=(100000, 63)):
    if len(data_shape) == 2:
        data_mat = np.memmap(data_dir + 'data_mat_' + whiten_mode + '_whitened.npy',
                             dtype=np.float32, mode='r', shape=data_shape)
        n_samples, n_features = data_shape
        idx = 0
        while True:
            if idx + batch_size < n_samples:
                batch = data_mat[idx:(idx + batch_size), :]
                idx += batch_size
            else:
                last_bit = data_mat[idx:, :]
                idx = (idx + batch_size) % n_samples
                first_bit = data_mat[:idx, :]
                batch = np.concatenate((last_bit, first_bit), axis=0)
            yield batch
    elif len(data_shape) == 3:
        data_mat = np.memmap(data_dir + 'data_mat_' + whiten_mode + '_channel_whitened.npy',
                             dtype=np.float32, mode='r', shape=data_shape)
        n_samples, n_channels, n_features = data_shape
        idx = 0
        while True:
            if idx + batch_size < n_samples:
                batch = data_mat[idx:(idx + batch_size), :, :]
                idx += batch_size
            else:
                last_bit = data_mat[idx:, :, :]
                idx = (idx + batch_size) % n_samples
                first_bit = data_mat[:idx, :, :]
                batch = np.concatenate((last_bit, first_bit), axis=0)
            yield batch
    else:
        raise NotImplementedError


def plot_img_mats(mat, color=False, rescale=False, show=True, save_path=''):
    """ plot l,m,n[,3] mats as l m by n gray-scale or color images """
    n = mat.shape[0]
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    if rescale:
        mat = (mat - np.min(mat)) / (np.max(mat) - np.min(mat))
    else:
        mat = np.maximum(mat, 0.0)
        mat = np.minimum(mat, 1.0)
    if not color:
        plt.style.use('grayscale')
    fig, ax_list = plt.subplots(ncols=cols, nrows=rows)
    ax_list = ax_list.flatten()

    for idx, ax in enumerate(ax_list):
        if idx >= n:
            ax.axis('off')
        else:
            if color:
                ax.imshow(mat[idx, :, :, :], interpolation='none')
            else:
                ax.imshow(mat[idx, :, :], interpolation='none')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    if save_path:
        plt.savefig(save_path, format='png')
    if show:
        plt.show()
    else:
        plt.close()


def fast_ica_comp():
    ph = 8
    pw = 8
    n_samples = 5000
    color = False
    # data = get_patches(n_samples, ph, pw, color=color)
    # data, rerotate = pca_whiten(data)
    data_gen = patch_batch_gen(n_samples)
    whiten, un_whiten = next(data_gen)
    data = next(data_gen)
    ica = FastICA(whiten=False, max_iter=1000)
    ica.fit(data)
    comps = ica.components_
    comps = np.dot(comps, un_whiten)
    comps -= np.min(comps)
    comps /= np.max(comps)
    plot_img_mats(np.reshape(comps, [-1, ph, pw]), color=color)


def get_optimizer(name, lr_pl, momentum=0.9):
    if name.lower() == 'adam':
        return tf.train.AdamOptimizer(lr_pl)
    elif name.lower() == 'momentum':
        return tf.train.MomentumOptimizer(lr_pl, momentum=momentum)
    elif name.lower() == 'adagrad':
        return tf.train.AdagradOptimizer(lr_pl)
    else:
        raise NotImplementedError