import os

import numpy as np
import tensorflow as tf
from tf_alexnet.alexnet import AlexNet
from tf_vgg.vgg16 import Vgg16
from utils.filehandling import load_image
from utils.whitening import pca_whiten_mats, zca_whiten_mats


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
                    cov_acc = cov_acc + (np.outer(map_patch, map_patch) / (num_patches - 1))

                    if idx + (count * batch_size) % (num_patches // 100) == 0:
                        print(100 * (idx + count * batch_size) / num_patches, '% cov accumulation done')

            cov = cov_acc
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
    ABOUT TO BE DEPRECATED
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


def make_channel_separate_patch_data(num_patches, ph, pw, classifier, map_name, n_channels,
                                     save_dir, whiten_mode='pca', batch_size=10,
                                     store_mean=True, channel_mean=True, store_cov=True):
    """
    creates whitening, covariance, raw and whitened feature matrices for separate channels.
    They are saved as 3d matrices where the first dimension is the channel index
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    raw_mat = raw_patch_data_mat(map_name, classifier, num_patches, ph, pw, batch_size, n_channels, save_dir)
    print('raw mat done')

    data_mat = normed_patch_data_mat(raw_mat, store_mean, channel_mean, store_cov, save_dir)
    print('normed mat done')

    cov = channel_independent_cov_acc(data_mat, save_dir)
    print('cov done')

    channel_whiten, channel_unwhiten = channel_independent_whitening_mats(cov, whiten_mode, save_dir)
    print('whitening mats done')


    data_mat = np.memmap(save_dir + 'white_mat_' + whiten_mode + '_channelwise.npy', dtype=np.float32, mode='w+',
                         shape=(num_patches, n_channels, channel_whiten.shape[1]))

    n_feats_per_channel = cov.shape[1]
    for idx in range(num_patches):
        image = data_mat[idx, :, :]  # [n_c, n_fpc]               .reshape([n_feats_per_channel, n_channels]).T
        image = np.expand_dims(image, axis=2)
        data_mat[idx, :] = np.squeeze(channel_whiten @ image)

    print('whitened data done')


def raw_patch_data_mat(map_name, classifier, num_patches, ph, pw, batch_size, n_channels,
                       save_dir, file_name='raw_mat.npy'):
    """
    create (num_patches, n_channels, feats per channel) matrix of extracted patches
    """
    assert num_patches % batch_size == 0

    if classifier.lower() == 'vgg16':
        classifier = Vgg16()
    elif classifier.lower() == 'alexnet':
        classifier = AlexNet()
    else:
        raise NotImplementedError

    file_path = save_dir + file_name

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

            raw_mat = np.memmap(file_path, dtype=np.float32, mode='w+',
                                shape=(num_patches, n_channels, n_feats_per_channel))

            max_h = map_dims[1] - ph
            max_w = map_dims[2] - pw

            with open(data_path + img_file) as f:
                image_files = [k.rstrip() for k in f.readlines()]

            image_paths = [data_path + 'images_resized/' + k[:-len('JPEG')] + 'bmp' for k in image_files]
            img_mat = np.zeros(shape=[batch_size, 224, 224, 3])

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
                    raw_mat[idx + (count * batch_size), :, :] = map_patch

        del raw_mat

    raw_mat = np.memmap(file_path, dtype=np.float32, mode='r',
                        shape=(num_patches, n_features))
    return raw_mat


def normed_patch_data_mat(raw_mat, store_mean, channel_mean, store_cov, save_dir, file_name='normed_mat.npy'):

    return raw_mat

def channel_independent_cov_acc(data_mat, save_dir, batch_size=100, file_name='cov.npy'):
    num_patches, n_channels, n_feats_per_channel = data_mat.shape

    channel_covs = np.zeros(shape=[n_channels, n_feats_per_channel, n_feats_per_channel])

    assert num_patches % batch_size == 0

    for idx in range(num_patches % batch_size):
        batch = data_mat[idx * batch_size:(idx + 1) * batch_size, :, :]  # batch has shape [bs, n_c, n_fpc]
        batch = np.rollaxis(batch, axis=0, start=2)
        channel_covs += np.matmul(np.rollaxis(batch, axis=0, start=2), np.rollaxis(batch, axis=0, start=1))

    channel_covs = channel_covs / (num_patches - 1)

    np.save(save_dir + file_name, channel_covs)

    return channel_covs


def channel_independent_whitening_mats(cov, whiten_mode, save_dir):
    n_channels, n_feats_per_channel = cov.shape[:2]

    channel_whiten = np.zeros(shape=[n_channels, n_feats_per_channel - 1, n_feats_per_channel])
    channel_unwhiten = np.zeros(shape=[n_channels, n_feats_per_channel - 1, n_feats_per_channel])

    for idx in range(n_channels):

        if whiten_mode == 'pca':
            whiten, unwhiten = pca_whiten_mats(cov[idx, :, :], n_to_drop=1)
        elif whiten_mode == 'zca':
            raise NotImplementedError
        else:
            raise NotImplementedError

        channel_whiten[idx, :, :] = whiten
        channel_unwhiten[idx, :, :] = unwhiten

    np.save(save_dir + 'whiten_' + whiten_mode + '_channelwise.npy', channel_whiten)
    np.save(save_dir + 'unwhiten_' + whiten_mode + '_channelwise.npy', channel_unwhiten)

    return channel_whiten, channel_unwhiten