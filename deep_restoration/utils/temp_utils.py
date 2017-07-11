import numpy as np
from utils.filehandling_utils import load_image
import os
import skimage
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt


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


def pca_whiten_mats(cov):
    e_vals, e_vecs = np.linalg.eigh(cov)  # seems like vals are sorted ascending, relying on that for now
    # print(e_vals)
    # e_vals += 0.000001
    # x, y = zip(sorted(zip(e_vals, range(e_vals.shape[0]))))
    # print(x)
    # print(y)
    e_vals = e_vals[1:]  # dismiss lowest eigenvalue, due to mean substraction
    e_vecs = e_vecs[:, 1:]
    sqrt_vals = np.sqrt(np.maximum(e_vals, 0))
    whiten = np.diag(1. / sqrt_vals) @ e_vecs.T
    un_whiten = e_vecs @ np.diag(sqrt_vals)
    print(whiten.shape)
    print(un_whiten.shape)
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


def make_data_mat_from_patches(data_dir='./data/patches_gray/8by8/', whiten_mode='pca'):
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


def make_cov_acc(data_dir='./data/patches_gray/8by8/'):
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


def patch_batch_gen(batch_size, data_dir='./data/patches_gray/new8by8/', whiten_mode='pca',
                    data_shape=(100000, 63)):
    data_mat = np.memmap(data_dir + 'data_mat_' + whiten_mode + '_whitened.npy', dtype=np.float32, mode='r',
                         shape=data_shape)
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


def plot_img_mats(mat, color=False):
    """ plot l*m*n mats as l m by n gray-scale images """
    n = mat.shape[0]
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n // cols))
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
    plt.show()


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