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
import tensorflow as tf
from tf_alexnet.alexnet import AlexNet
from tf_vgg.vgg16 import Vgg16
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DATA_PATH = '../data/imagenet2012-validationset/'
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


def load_image(path, res=(224, 224), resize=False):
    """
    taken from https://github.com/machrisaa/tensorflow-vgg/utils.py
    returns image of shape [res[0], res[1], 3]
    [height, width, depth]
    """
    if path.endswith('npy'):
        assert resize is False
        return np.load(path)

    img = skimage.io.imread(path)
    if img.shape[2] == 4:
        img = skimage.color.rgba2rgb(img, [0, 0, 0])

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


def resize_all_images(target_resolution, target_dir, img_file='images.txt', source_dir='images'):
    with open(DATA_PATH + img_file) as f:
        image_files = [k.rstrip() for k in f.readlines()]
        count = 0
        for img_file in image_files:
            target_file = DATA_PATH + target_dir + '/' + img_file[:-len('JPEG')] + 'bmp'
            if not os.path.isfile(target_file):
                image = load_image(DATA_PATH + source_dir + '/' + img_file, res=target_resolution, resize=True)
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


def correctly_classified_validation_subset(classifier='alexnet', batch_size=50, topx=1):
    if classifier.lower() == 'vgg16':
        model = Vgg16()
        image_subdir = 'images_resized_224/'
        img_dims = [batch_size, 224, 224, 3]
    elif classifier.lower() == 'alexnet':
        model = AlexNet()
        image_subdir = 'images_resized_227/'
        img_dims = [batch_size, 227, 227, 3]
    else:
        raise ValueError

    image_set_file = 'validate_2k_images.txt'
    label_keys = 'validate_2k_labels.txt'
    
    # compute actual labels
    with open(DATA_PATH + label_keys) as f:
        labels = label_strings_to_ints([k.rstrip() for k in f.readlines()])
    
    with open(DATA_PATH + image_set_file) as f:
        image_files = [k.rstrip()[:-len('JPEG')] + 'bmp' for k in f.readlines()]
        image_files = [DATA_PATH + image_subdir + k for k in image_files]
    assert len(image_files) % batch_size == 0

    image_batches = [image_files[k:k+batch_size] for k in range(len(image_files) // batch_size)]
    label_batches = [labels[k:k+batch_size] for k in range(len(labels) // batch_size)]
    matches = []

    with tf.Graph().as_default() as graph:
        img_pl = tf.placeholder(dtype=tf.float32, shape=img_dims)
        model.build(img_pl)

        softmax = graph.get_tensor_by_name('softmax:0')

        with tf.Session() as sess:
            for batch_paths, batch_labels in zip(image_batches, label_batches):
                batch_mat = np.asarray([load_image(k) for k in batch_paths])
                pred = sess.run(softmax, feed_dict={img_pl: batch_mat})

                pred_min = np.min(pred)
                batch_matches = [False] * batch_size
                for count in range(topx):
                    pred_labels = list(np.argmax(pred, axis=1))
                    pred_matches = [k[0] == k[1] for k in zip(pred_labels, batch_labels)]
                    batch_matches = [k[0] or k[1] for k in zip(batch_matches, pred_matches)]

                    max_ids = [(k[0], k[1]) for k in enumerate(pred_labels)]
                    pred[list(zip(*max_ids))] = pred_min - 1
                matches.extend(batch_matches)
    correct_images = [image_files[k] + '\n' for k in range(len(image_files)) if matches[k]]
    print(len(correct_images))
    with open(DATA_PATH + '{}_val_2k_top{}_correct.txt'.format(classifier, topx), mode='w') as f:
        f.writelines(correct_images)


def label_strings_to_ints(label_str_list):
    with open(DATA_PATH + 'label_names.txt') as f:
        label_dict = dict()
        for idx, label in enumerate([k.split(' ')[0] for k in f.readlines()]):
            label_dict[label] = idx

    return [label_dict[l] for l in label_str_list]


def prepare_scalar_logs(path):
    size_guidance = {'compressedHistograms': 1, 'images': 1, 'audio': 1, 'scalars': 0, 'histograms': 1, 'tensors': 1}
    event_acc = EventAccumulator(path, size_guidance=size_guidance)
    event_acc.Reload()
    scalar_logs = dict()
    for tag in event_acc.Tags()['scalars']:
        events = event_acc.Scalars(tag)
        steps = [k.step for k in events]
        values = [k.value for k in events]
        scalar_logs[tag] = (steps, values)
    return scalar_logs


def plot_opt_inv_experiment(path, exp_subdirs, log_tags):
    exp_logs = dict()
    for exp in exp_subdirs:
        exp_path = os.path.join(path, exp_subdirs[exp], 'summaries')
        print(exp_path)
        exp_logs[exp] = prepare_scalar_logs(exp_path)

    for log_name in log_tags:
        tag = log_tags[log_name]
        plt.figure()
        plt.title(log_name)
        for exp in exp_logs:
            log = exp_logs[exp]
            print(log, tag, exp_logs)
            if tag in log:
                steps, values = log[tag]
                plt.plot(steps, values, label=exp)
        plt.legend()
        plt.show()
        plt.close()


def plot_example_exp():
    path = '../logs/opt_inversion/alexnet/slim_vs_img/c2l_to_c1l'
    exp_subdirs = {'No prior': 'no_prior',
                   'Pre-image with prior': 'pre_image_8x8_full_prior/1e-3',
                   'Pre-image with no prior': 'pre_image_no_prior'}
    log_tags = {'Total loss': 'Total_Loss',
                'Reconstruction error': 'MSE_Reconstruction_1'}
    plot_opt_inv_experiment(path, exp_subdirs, log_tags)