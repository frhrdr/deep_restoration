import matplotlib
matplotlib.use('tkagg', force=True)
import xml.etree.ElementTree as ET
from collections import defaultdict
import skimage
import skimage.io
import skimage.transform
from skimage.color import grey2rgb
import time
import warnings
import os
import numpy as np
import matplotlib.pyplot as plt


DATA_PATH = './data/imagenet2012-validationset/'
# DATA_PATH = './data/selected/'


def get_labels(xml_data_dir='xml_annotations/', out_file='labels.txt', names_file='images.txt'):
    with open(DATA_PATH + names_file) as f:
        names = [k.rstrip() for k in f.readlines()]

    labels = []
    for name in names:
        xml_file = DATA_PATH + xml_data_dir + name.split('.')[0] + '.xml'
        root = ET.parse(xml_file).getroot()
        label = root.find('./object/name').text
        labels.append(label)

    with open(DATA_PATH + out_file, 'w') as f:
        f.writelines([k + '\n' for k in labels])


def count_labels(label_file='labels.txt'):
    with open(DATA_PATH + label_file) as f:
        labels = [k.rstrip() for k in f.readlines()]
    counter = defaultdict(int)
    for label in labels:
        counter[label] += 1
    for key in counter:
        print(key, counter[key])


def make_balanced_subset(names_file='images.txt', labels_file='labels.txt', num_per_label=1, cut_off=None):
    with open(DATA_PATH + names_file) as f:
        names = [k.rstrip() for k in f.readlines()]
    with open(DATA_PATH + labels_file) as f:
        labels = [k.rstrip() for k in f.readlines()]

    counter = defaultdict(int)
    subset = []
    for idx, (name, label) in enumerate(zip(names, labels)):
        if counter[label] < num_per_label:
            subset.append([name, label])
            counter[label] += 1

    if cut_off is not None:
        subset = subset[:cut_off]
        id_string = 'subset_cutoff_' + str(cut_off) + '_'
    else:
        id_string = 'subset_' + str(num_per_label) + 'k_'
    s_names, s_labels = zip(*subset)

    with open(DATA_PATH + id_string + 'images.txt', 'w') as f:
        f.writelines([k + '\n' for k in s_names])
    with open(DATA_PATH + id_string + 'labels.txt', 'w') as f:
        f.writelines([k + '\n' for k in s_labels])


def get_feature_files(layer_name, subset_file):
    with open(DATA_PATH + subset_file) as f:
        names = [k.split('.')[0] for k in f.readlines()]
        paths = ['./data/features/' + layer_name + '/' + layer_name + '_' + k + '.npy' for k in names]
    return paths


def find_broken_files(names_file='images.txt'):
    with open(DATA_PATH + names_file) as f:
        image_files = [k.rstrip() for k in f.readlines()]

    image_paths = [DATA_PATH + 'images/' + k for k in image_files]
    for p in image_paths:
        try:
            skimage.io.imread(p)
        except ValueError:
            print('broken/unreadable file: ' + p)


def get_img_files_complement(file_a='images.txt', file_b='validate_2k_images.txt',
                             complement_name='train_48k_images.txt'):
    with open(DATA_PATH + file_a) as f:
        lines_a = [k.rstrip() for k in f.readlines()]
    with open(DATA_PATH + file_b) as f:
        lines_b = [k.rstrip() for k in f.readlines()]
    comp = list(set(lines_a) - set(lines_b))

    print('complement has ' + str(len(comp)) + ' elements')
    with open(DATA_PATH + complement_name, 'w') as f:
        f.writelines([k + '\n' for k in comp])


def load_image(path, res=(224, 224), resize=True):
    """
    taken from https://github.com/machrisaa/tensorflow-vgg/utils.py
    returns image of shape [res[0], res[1], 3]
    [height, width, depth]
    """
    img = skimage.io.imread(path)
    if resize:
        # noinspection PyUnresolvedReferences
        assert (0 <= img).all() and (img <= 255.0).all()
        # we crop image from center
        short_edge = min(img.shape[:2])
        yy = int((img.shape[0] - short_edge) / 2)
        xx = int((img.shape[1] - short_edge) / 2)
        crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
        resized_img = skimage.transform.resize(crop_img, res, mode='constant')
        return resized_img
    else:
        return img


def save_dict(d, file_path):
    lines = []
    for key in d.keys():
        lines.append(key + ' ' + str(d[key]) + '\n')
    with open(file_path, mode='w') as f:
        f.writelines(lines)


def resize_all_images(img_file='images.txt', source_dir='images', target_dir='images_resized'):
    with open(DATA_PATH + img_file) as f:
        image_files = [k.rstrip() for k in f.readlines()]
        count = 0
        for img_file in image_files:
            target_file = DATA_PATH + target_dir + '/' + img_file[:-len('JPEG')] + 'bmp'
            if not os.path.isfile(target_file):
                image = load_image(DATA_PATH + source_dir + '/' + img_file, resize=True)
                if len(image.shape) == 2:
                    image = grey2rgb(image)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    skimage.io.imsave(target_file, image)
            count += 1
            if count % 100 == 0:
                print(count)


def measure_format_access_speeds():
    count = range(1000)

    t = time.time()
    for c in count:
        image = load_image(DATA_PATH + 'images_resized/ILSVRC2012_val_00000001' + '.bmp', resize=False)
    print('bmp: ' + str(time.time() - t))

    t = time.time()
    for c in count:
        image = load_image(DATA_PATH + 'images_resized/ILSVRC2012_val_00000001' + '.png', resize=False)
    print('png: ' + str(time.time() - t))

    t = time.time()
    for c in count:
        image = load_image(DATA_PATH + 'images/ILSVRC2012_val_00000001' + '.JPEG', resize=True)
    print('jpg: ' + str(time.time() - t))


def concat_mv_images(num=21):
    model = 'vgg16'
    w = 7
    h = 3
    img_mat = np.zeros(shape=(h * 224, w * 224, 4))
    for idx in range(num):
        img = skimage.io.imread('./logs/mahendran_vedaldi/' + model + '/l{0}/rec_2000.png'.format(idx + 1))
        img_mat[224 * (idx // w):224 * (idx // w + 1),
                224 * (idx % w): 224 * (idx % w + 1), :] = img[:, 224:, :]
    path = './logs/mahendran_vedaldi/' + model + '/mh_' + model + '_overview2.png'
    img_mat /= 255.0
    fig = plt.figure(frameon=False)
    fig.set_size_inches(w, h)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img_mat, aspect='auto')
    plt.savefig(path, format='png', dpi=224)
    plt.close()


def make_patches_data(num_patches, ph=8, pw=8, color=False, save_dir='./data/patches_color/8x8/', whiten_mode='pca'):
    img_hw = 224
    max_h = img_hw - ph
    max_w = img_hw - pw
    data_path = './data/imagenet2012-validationset/'
    img_file = 'train_48k_images.txt'

    mm = np.memmap(save_dir + '/data_mat_' + whiten_mode + '.npy', dtype=np.float32, mode='w+',
                   shape=(num_patches, 63))
    with open(data_path + img_file) as f:
        image_files = [k.rstrip() for k in f.readlines()]

    image_paths = [data_path + 'images_resized/' +
                   k[:-len('JPEG')] + 'bmp' for k in image_files]

    for idx in range(num_patches):
        img_path = image_paths[idx % len(image_paths)]
        h = np.random.randint(0, max_h)
        w = np.random.randint(0, max_w)
        image = load_image(img_path, resize=False)
        image = image.astype(float)
        image = image[h:h + ph, w:w + pw, :]
        if not color:
            image = skimage.color.rgb2gray(image)
        # image = image.flatten()
        image -= 0.01  # to avoid max > 1
        image /= 255.0  # map to range [0,1]

        # image -= image.mean()  # subtract image mean
        target_file = save_dir + 'patch_' + str(idx) + '.bmp'
        skimage.io.imsave(target_file, image)


def mat_to_img(mat_file, rescale=True, cols=1, rows=1):
    plot_mat = np.load(mat_file)
    plot_mat = np.reshape(plot_mat, [224, 224, 3])
    if rescale:
        plot_mat = (plot_mat - np.min(plot_mat)) / (np.max(plot_mat) - np.min(plot_mat))  # M&V just rescale
    fig = plt.figure(frameon=False)
    fig.set_size_inches(cols, rows)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(plot_mat, aspect='auto')
    plt.savefig(mat_file[:-len('npy')] + 'png',
                format='png', dpi=224)
    plt.close()