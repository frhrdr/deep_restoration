import xml.etree.ElementTree as ET
from collections import defaultdict
import skimage
import skimage.io
import skimage.transform


data_path = './data/imagenet2012-validationset/'


def get_labels(xml_data_dir='xml_annotations/', out_file='val_labels.txt', names_file='val_images.txt'):
    with open(data_path + names_file) as f:
        names = [k.rstrip() for k in f.readlines()]

    labels = []
    for name in names:
        xml_file = data_path + xml_data_dir + name.split('.')[0] + '.xml'
        root = ET.parse(xml_file).getroot()
        label = root.find('./object/name').text
        labels.append(label)

    with open(data_path + out_file, 'w') as f:
        f.writelines([k + '\n' for k in labels])


def count_labels(label_file='val_labels.txt'):
    with open(data_path + label_file) as f:
        labels = [k.rstrip() for k in f.readlines()]
    counter = defaultdict(int)
    for label in labels:
        counter[label] += 1
    for key in counter:
        print(key, counter[key])


def make_balanced_subset(names_file='val_images.txt', labels_file='val_labels.txt', num_per_label=1, cut_off=None):
    with open(data_path + names_file) as f:
        names = [k.rstrip() for k in f.readlines()]
    with open(data_path + labels_file) as f:
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

    with open(data_path + id_string + 'images.txt', 'w') as f:
        f.writelines([k + '\n' for k in s_names])
    with open(data_path + id_string + 'labels.txt', 'w') as f:
        f.writelines([k + '\n' for k in s_labels])


def get_feature_files(layer_name, subset_file):
    with open(data_path + subset_file) as f:
        names = [k.split('.')[0] for k in f.readlines()]
        paths = ['./data/features/' + layer_name + '/' + layer_name + '_' + k + '.npy' for k in names]
    return paths


def find_broken_files(names_file='val_images.txt'):
    with open(data_path + names_file) as f:
        image_files = [k.rstrip() for k in f.readlines()]

    image_paths = [data_path + 'images/' + k for k in image_files]
    for p in image_paths:
        try:
            skimage.io.imread(p)
        except ValueError:
            print('broken/unreadable file: ' + p)


# taken from https://github.com/machrisaa/tensorflow-vgg/utils.py
# returns image of shape [res[0], res[1], 3]
# [height, width, depth]
def load_image(path, res=(224, 224)):
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, res, mode='constant')
    return resized_img