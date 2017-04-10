import xml.etree.ElementTree as ET
from collections import defaultdict

data_path = '/data/imagenet2012-validationset/'


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


def make_balanced_subset(names_file='val_images.txt', labels_file='val_labels.txt', num_per_label=2):
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
    s_names, s_labels = zip(*subset)

    with open(data_path + 'subset_' + str(num_per_label) + 'k_names.txt', 'w') as f:
        f.writelines([k + '\n' for k in s_names])
    with open(data_path + 'subset_' + str(num_per_label) + 'k_labels.txt', 'w') as f:
        f.writelines([k + '\n' for k in s_labels])

count_labels()