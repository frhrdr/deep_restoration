import numpy as np
import tensorflow as tf
from skimage.io import imsave
from utils.filehandling import load_image
from tf_alexnet.alexnet import AlexNet


def get_orig_featmaps():
    with tf.Graph().as_default() as graph:
        img = load_image('/home/frederik/PycharmProjects/deep_restoration/data/selected/images_resized_227/red-fox.bmp')
        image = tf.constant(img, dtype=tf.float32, shape=[1, 227, 227, 3])
        net = AlexNet()
        net.build(image, rescale=1.0)
        feat_map = graph.get_tensor_by_name('conv1/lin' + ':0')
        with tf.Session() as sess:
            feat_map_mat = sess.run(feat_map)
        return feat_map_mat


def make_featmap_plot():
    base_path = '/home/frederik/PycharmProjects/deep_restoration/logs/opt_inversion/alexnet/c2l_to_c1l/'
    # base_path = '../logs/opt_inversion/alexnet/c2l_to_c1l/'
    add_paths = [('full', 'full_prior/'), ('slim', 'slim_prior/'), ('dual', 'dual_prior/'),
                 ('chan', 'chan_prior/adam/'), ('tv', 'tv_prior/')]
    file_name = 'run_final/mats/rec_20000.npy'

    max_n_featmaps_to_plot = 10

    mats = []
    orig = get_orig_featmaps()
    print(orig.shape)
    mats.append(orig[:, ...])

    for k, p in add_paths:
        path = base_path + p + file_name
        mat = np.load(path)
        mats.append(mat)

    collages = []
    for mat in mats:
        print(mat.shape)
        _, height, width, n_channels = mat.shape
        n_featmaps_to_plot = min([max_n_featmaps_to_plot, n_channels])

        collage = np.zeros(shape=(width, n_featmaps_to_plot * height))
        for idy in range(n_featmaps_to_plot):
            collage[:, idy * height:(idy + 1) * height] = mat[:, :, :, idy]
        collages.append(collage)

    plot_mats = np.concatenate(collages, axis=0)

    for nf in range(n_featmaps_to_plot):
        col = plot_mats[:, nf * width:(nf + 1) * width]
        col = (col - np.min(col)) / (np.max(col) - np.min(col))
        plot_mats[:, nf * width:(nf + 1) * width] = col

    print(plot_mats.shape)
    imsave('./c1l_featmap_vis.png', plot_mats)

make_featmap_plot()