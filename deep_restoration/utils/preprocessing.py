import os

import numpy as np
import tensorflow as tf
from tf_alexnet.alexnet import AlexNet
from tf_vgg.vgg16 import Vgg16
from utils.filehandling import load_image
from utils.whitening import pca_whiten_mats, zca_whiten_mats


def dep_make_feat_map_mats(num_patches, map_name, classifier='alexnet', ph=8, pw=8,
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


def dep_make_reduced_feat_map_mats(num_patches, load_dir, n_features, n_to_keep,
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


def dep_make_channel_separate_feat_map_mats(num_patches, ph, pw, classifier, map_name, n_channels,
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


def make_flattened_patch_data(num_patches, ph, pw, classifier, map_name, n_channels,
                              save_dir, n_feats_white, whiten_mode='pca', batch_size=100,
                              mean_mode='local_full', sdev_mode='global_feature',
                              raw_mat_load_path=''):
    """
    creates whitening, covariance, raw and whitened feature matrices for separate channels.
    all data is saved as [n_patches, n_channels, n_features_per_channel]
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if raw_mat_load_path:
        raw_mat = np.memmap(raw_mat_load_path, dtype=np.float32, mode='r',
                            shape=(num_patches, n_channels, ph * pw))
    else:
        raw_mat = raw_patch_data_mat(map_name, classifier, num_patches, ph, pw, batch_size, n_channels, save_dir)
    print('raw mat done')

    norm_mat = normed_patch_data_mat(raw_mat, save_dir, mean_mode=mean_mode, sdev_mode=sdev_mode)
    print('normed mat done')

    print('mat dims pre flatten:', norm_mat.shape)
    flat_mat = norm_mat.reshape([num_patches, -1])

    cov = flattened_cov_acc(flat_mat, save_dir)
    print('cov done')

    whiten, unwhiten = flattened_whitening_mats(cov, whiten_mode, save_dir, n_feats_white)
    print('whitening mats done')

    data_mat = np.memmap(save_dir + 'data_mat_' + whiten_mode + '_whitened.npy', dtype=np.float32, mode='w+',
                         shape=(num_patches, n_feats_white))

    for idx in range(num_patches // batch_size):
        image = flat_mat[idx * batch_size:(idx + 1) * batch_size, :]  # [bs, n_f]
        # whiten is [n_fw, n_f], target [bs, n_fw]
        data_mat[idx * batch_size:(idx + 1) * batch_size, :] = image @ whiten.T  # [bs, n_f] x [n_f, n_fw] = [bs, n_fw]
    print('whitened data done')


def make_channel_separate_patch_data(num_patches, ph, pw, classifier, map_name, n_channels,
                                     save_dir, whiten_mode='pca', batch_size=100,
                                     mean_mode='global_channel', sdev_mode='global_channel',
                                     raw_mat_load_path=''):
    """
    creates whitening, covariance, raw and whitened feature matrices for separate channels.
    They are saved as 3d matrices where the first dimension is the channel index
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if raw_mat_load_path:
        raw_mat = np.memmap(raw_mat_load_path, dtype=np.float32, mode='r',
                            shape=(num_patches, n_channels, ph * pw))
    else:
        raw_mat = raw_patch_data_mat(map_name, classifier, num_patches, ph, pw, batch_size, n_channels, save_dir)
    print('raw mat done')

    norm_mat = normed_patch_data_mat(raw_mat, save_dir, mean_mode=mean_mode, sdev_mode=sdev_mode)
    print('normed mat done')

    cov = channel_independent_cov_acc(norm_mat, save_dir)
    print('cov done')

    channel_whiten, channel_unwhiten = channel_independent_whitening_mats(cov, whiten_mode, save_dir)
    print('whitening mats done')

    data_mat = np.memmap(save_dir + 'data_mat_' + whiten_mode + '_whitened_channelwise.npy', dtype=np.float32, mode='w+',
                         shape=(num_patches, n_channels, channel_whiten.shape[1]))

    for idx in range(num_patches // batch_size):
        image = norm_mat[idx * batch_size:(idx + 1) * batch_size, :, :]  # [bs, n_c, n_fpc]
        # channel_whiten is [n_c, n_fpcw, n_fpc], target [bs, n_c, n_fpcw]
        image = np.expand_dims(image, axis=3)  # [bs, n_c, n_fpc, 1]
        data_mat[idx * batch_size:(idx + 1) * batch_size, :, :] = np.squeeze(channel_whiten @ image)  # [n_c, n_fpcw, n_fpc] x [n_c, n_fpc, 1] = [n_c, n_fpcw]
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
            # n_features = n_feats_per_channel * map_dims[3]

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
                    map_patch = np.transpose(map_mat[idx, h:h + ph, w:w + pw, :], axes=(2, 0, 1))
                    map_patch = map_patch.reshape([n_channels, -1]).astype(np.float32)
                    raw_mat[idx + (count * batch_size), :, :] = map_patch

        del raw_mat

    raw_mat = np.memmap(file_path, dtype=np.float32, mode='r',
                        shape=(num_patches, n_channels, n_feats_per_channel))
    return raw_mat


def normed_patch_data_mat(raw_mat, save_dir,
                          mean_mode='global_channel', sdev_mode='global_channel',
                          file_name='normed_mat.npy', batch_size=0):

    modes =  ('global_channel', 'global_feature', 'local_channel', 'local_full',
              'gc', 'gf', 'lc', 'lf', 'none')
    assert mean_mode in modes
    assert sdev_mode in modes or isinstance(sdev_mode, float)
    num_patches, n_channels, n_feats_per_channel = raw_mat.shape
    batch_size = batch_size if batch_size else num_patches
    assert num_patches % batch_size == 0
    data_mat = np.memmap(save_dir + file_name, dtype=np.float32, mode='w+',
                         shape=raw_mat.shape)

    ###### MEAN treatment ######
    if mean_mode in ('global_channel', 'gc'):
        channel_mean = np.mean(raw_mat, axis=(0, 2))
        for idx in range(num_patches // batch_size):
            batch = raw_mat[idx * batch_size:(idx + 1) * batch_size, :, :]
            batch_t = np.transpose(batch, axes=(0, 2, 1))
            batch_t_centered = batch_t - channel_mean
            batch_centered = np.transpose(batch_t_centered, axes=(0, 2, 1))
            data_mat[idx * batch_size:(idx + 1) * batch_size, :, :] = batch_centered

        np.save(save_dir + 'data_mean.npy', channel_mean)

    elif mean_mode in ('global_feature', 'gf'):
        feature_mean= np.mean(raw_mat, axis=0)

        for idx in range(num_patches // batch_size):
            batch = raw_mat[idx * batch_size:(idx + 1) * batch_size, :, :]
            data_mat[idx * batch_size:(idx + 1) * batch_size, :, :] = batch - feature_mean

        np.save(save_dir + 'data_sdev.npy', feature_mean)

    elif mean_mode in ('local_channel', 'lc'):
        for idx in range(num_patches // batch_size):
            batch = raw_mat[idx * batch_size:(idx + 1) * batch_size, :, :]
            channel_mean = np.mean(batch, axis=2)  # shape=[n_patches, n_channels]
            batch_t = np.transpose(batch, axes=(2, 0, 1))
            batch_t_centered = batch_t - channel_mean
            batch_centered = np.transpose(batch_t_centered, axes=(1, 2, 0))
            data_mat[idx * batch_size:(idx + 1) * batch_size, :, :] = batch_centered

    elif mean_mode in ('local_full', 'lf'):
        for idx in range(num_patches // batch_size):
            batch = raw_mat[idx * batch_size:(idx + 1) * batch_size, :, :]
            sample_mean = np.mean(batch, axis=(1, 2))
            batch_t = np.transpose(batch)
            batch_t_centered = batch_t - sample_mean
            batch_centered = np.transpose(batch_t_centered)
            data_mat[idx * batch_size:(idx + 1) * batch_size, :, :] = batch_centered

    else:  # mean_mode is 'none'
        pass

    ###### SDEV treatment ######
    if sdev_mode in ('global_channel', 'gc'):
        feat_sdev = np.std(raw_mat, axis=0)
        channel_sdev = np.mean(feat_sdev, axis=1)
        for idx in range(num_patches // batch_size):
            batch = raw_mat[idx * batch_size:(idx + 1) * batch_size, :, :]
            batch = np.rollaxis(np.rollaxis(batch, axis=2, start=1) / channel_sdev, axis=2, start=1)
            data_mat[idx * batch_size:(idx + 1) * batch_size, :, :] = batch

        np.save(save_dir + 'data_sdev.npy', channel_sdev)

    elif sdev_mode in ('global_feature', 'gf'):
        feat_sdev = np.std(raw_mat, axis=0)

        for idx in range(num_patches // batch_size):
            batch = raw_mat[idx * batch_size:(idx + 1) * batch_size, :, :]
            batch = batch / feat_sdev
            data_mat[idx * batch_size:(idx + 1) * batch_size, :, :] = batch

        np.save(save_dir + 'data_sdev.npy', feat_sdev)


    elif sdev_mode in ('local_channel', 'lc'):  # seems like a bad idea anyway
        raise NotImplementedError

    elif sdev_mode in ('local_full', 'lf'):  # this too
        for idx in range(num_patches // batch_size):
            batch = raw_mat[idx * batch_size:(idx + 1) * batch_size, :, :]
            sample_sdev = np.std(batch, axis=(1, 2))
            batch_t = np.transpose(batch)
            batch_t_scaled = batch_t / sample_sdev
            batch_scaled = np.transpose(batch_t_scaled)
            data_mat[idx * batch_size:(idx + 1) * batch_size, :, :] = batch_scaled

    elif isinstance(sdev_mode, float):
        data_mat[:, :, :] = sdev_mode * raw_mat[:, :, :]

    else:  # sdev_mode == 'none'
        pass

    del data_mat
    data_mat = np.memmap(save_dir + file_name, dtype=np.float32, mode='r',
                         shape=raw_mat.shape)
    return data_mat


def flattened_cov_acc(norm_mat, save_dir, batch_size=100, file_name='cov.npy'):
    num_patches, n_feats = norm_mat.shape
    assert num_patches % batch_size == 0

    channel_covs = np.zeros(shape=[n_feats, n_feats])

    for idx in range(num_patches // batch_size):
        batch = norm_mat[idx * batch_size:(idx + 1) * batch_size, :]  # batch has shape [bs, n_f]
        channel_covs += np.matmul(batch.T, batch)

    channel_covs = channel_covs / (num_patches - 1)

    np.save(save_dir + file_name, channel_covs)
    return channel_covs


def channel_independent_cov_acc(norm_mat, save_dir, batch_size=100, file_name='cov.npy'):
    num_patches, n_channels, n_feats_per_channel = norm_mat.shape

    channel_covs = np.zeros(shape=[n_channels, n_feats_per_channel, n_feats_per_channel])

    assert num_patches % batch_size == 0

    for idx in range(num_patches // batch_size):
        batch = norm_mat[idx * batch_size:(idx + 1) * batch_size, :, :]  # batch has shape [bs, n_c, n_fpc]
        channel_covs += np.matmul(batch.transpose((1, 2, 0)), batch.transpose((1, 0, 2)))

    channel_covs = channel_covs / (num_patches - 1)

    np.save(save_dir + file_name, channel_covs)

    return channel_covs


def flattened_whitening_mats(cov, whiten_mode, save_dir, n_feats_white):
    n_to_drop = cov.shape[0] - n_feats_white
    if whiten_mode == 'pca':
        whiten, unwhiten = pca_whiten_mats(cov, n_to_drop=n_to_drop)
    elif whiten_mode == 'zca':
        raise NotImplementedError
    else:
        raise NotImplementedError

    np.save(save_dir + 'whiten_' + whiten_mode + '.npy', whiten)
    np.save(save_dir + 'unwhiten_' + whiten_mode + '.npy', unwhiten)

    return whiten, unwhiten


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

    np.save(save_dir + 'whiten_' + whiten_mode + '.npy', channel_whiten)
    np.save(save_dir + 'unwhiten_' + whiten_mode + '.npy', channel_unwhiten)

    return channel_whiten, channel_unwhiten


def preprocess_tensor(patch_tensor, mean_mode, sdev_mode):
    """
    to be called on a TensorFlow graph. takes patch tensor and loads means and sdevs.
    then applies the same preprocessing as used on the training set (whitening is done after)
    takes in tensor with n_channels as last dimension. outputs tensor with n_feats_per_channel as last dimension.

    :param patch_tensor: tensor of shape [n_patches, n_feats_per_channel, n_channels] to be processed
    :param mean_mode: mode for centering
    :param sdev_mode: mode for rescaling
    :param load_dir: if provided, npy mats are loaded from this directory.  # not needed!!
    otherwise, values must be loaded from checkpoints
    :return: normalized tensor of shape [n_patches, n_channels, n_feats_per_channel]
    """
    modes = ('global_channel', 'global_feature', 'local_channel', 'local_full',
             'gc', 'gf', 'lc', 'lf',
             'none')
    assert mean_mode in modes
    assert sdev_mode in modes or isinstance(sdev_mode, float)
    init_list = []
    n_patches, n_feats_per_channel, n_channels = [k.value for k in patch_tensor.get_shape()]
    print('seeing tensor shape as n_p={}, n_fpc={}, n_c={}'.format(n_patches, n_feats_per_channel, n_channels))
    n_feats_raw = n_feats_per_channel * n_channels

    ###### MEAN treatment ######
    if mean_mode in ('global_channel', 'gc'):
        mean_tensor = tf.get_variable('centering_mean', shape=(n_channels,), trainable=False, dtype=tf.float32)
        init_list.append(mean_tensor)
        patch_tensor = patch_tensor - mean_tensor

    elif mean_mode in ('global_feature', 'gf'):
        mean_tensor = tf.get_variable('centering_mean', shape=(n_feats_raw,), trainable=False, dtype=tf.float32)
        init_list.append(mean_tensor)
        patch_tensor = tf.reshape(patch_tensor, [n_patches, n_feats_raw]) - mean_tensor
        patch_tensor = tf.reshape(patch_tensor, shape=(n_patches, n_feats_per_channel, n_channels))

    elif mean_mode in ('local_channel', 'lc'):
        channel_mean = tf.reduce_mean(patch_tensor, axis=1)
        patch_tensor = tf.transpose(patch_tensor, perm=(1, 0, 2))
        patch_tensor = patch_tensor - channel_mean
        patch_tensor = tf.transpose(patch_tensor, perm=(1, 0, 2))

    elif mean_mode in ('local_full', 'lf'):
        patch_tensor = tf.reshape(patch_tensor, shape=(n_patches, n_feats_raw))
        sample_mean = tf.reduce_mean(patch_tensor, axis=1)
        patch_tensor = tf.transpose(patch_tensor, perm=(1, 0))
        patch_tensor = patch_tensor - sample_mean
        patch_tensor = tf.transpose(patch_tensor, perm=(1, 0))
        patch_tensor = tf.reshape(patch_tensor, shape=(n_patches, n_feats_per_channel, n_channels))

    else:  # mean_mode == 'none'
        raise NotImplementedError

    ###### SDEV treatment ######
    if sdev_mode in ('global_channel', 'gc'):
        sdev_tensor = tf.get_variable('rescaling_sdev', shape=(n_channels,), trainable=False, dtype=tf.float32)
        init_list.append(sdev_tensor)
        patch_tensor = patch_tensor / sdev_tensor

    elif sdev_mode in ('global_feature', 'gf'):
        sdev_tensor = tf.get_variable('rescaling_sdev', shape=(n_feats_raw,), trainable=False, dtype=tf.float32)
        init_list.append(sdev_tensor)
        patch_tensor = tf.reshape(patch_tensor, [n_patches, n_feats_raw]) / sdev_tensor
        patch_tensor = tf.reshape(patch_tensor, shape=(n_patches, n_feats_per_channel, n_channels))

    elif sdev_mode in ('local_channel', 'lc'):
        raise NotImplementedError

    elif sdev_mode in ('local_full', 'lf'):
        patch_tensor = tf.reshape(patch_tensor, shape=(n_patches, n_feats_raw))
        _, sample_sdev = tf.nn.moments(patch_tensor, axes=1)
        patch_tensor = tf.transpose(patch_tensor, perm=(1, 0))
        patch_tensor = patch_tensor / sample_sdev
        patch_tensor = tf.transpose(patch_tensor, perm=(1, 0))
        patch_tensor = tf.reshape(patch_tensor, shape=(n_patches, n_feats_per_channel, n_channels))

    elif isinstance(sdev_mode, float):
        patch_tensor *= tf.constant(sdev_mode, dtype=tf.float32, shape=[])

    else:  # sdev_mode == 'none'
        pass

    patch_tensor = tf.transpose(patch_tensor, perm=(0, 2, 1))
    return patch_tensor, init_list