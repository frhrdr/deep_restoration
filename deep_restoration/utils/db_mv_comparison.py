import numpy as np
import skimage.io
import os.path
from net_inversion import NetInversion
from modules.inv_default_modules import get_stacked_module
from utils.filehandling import load_image


def subset10_paths(classifier):
    assert classifier in ('alexnet', 'vgg16')
    img_hw = 227 if classifier == 'alexnet' else 224
    subset_dir = '../data/selected/images_resized_{}/'.format(img_hw)
    img_ids = (53, 76, 81, 99, 106, 108, 129, 153, 157, 160)
    subset_paths = ['{}val{}.bmp'.format(subset_dir, i) for i in img_ids]
    return subset_paths


def load_and_stack_imgs(img_paths):
    return np.stack([load_image(p) for p in img_paths], axis=0)


def cnn_inv_log_path(classifier, start_layer, rec_layer):
    return '../logs/cnn_inversion/{}/stack_{}_to_{}/'.format(classifier, start_layer, rec_layer)


def run_stacked_module(classifier, start_layer, rec_layer, use_solotrain=False,
                       subdir_name=None, retrieve_special=None):

    subdir_name = subdir_name or '{}_stack_{}_to_{}'.format(classifier, start_layer, rec_layer)
    alt_load_subdir = 'solotrain' if use_solotrain else subdir_name
    module_list = get_stacked_module(classifier, start_layer, rec_layer, alt_load_subdir=alt_load_subdir,
                                     subdir_name=subdir_name, trainable=False)
    save_subdir = 'stacked/' if use_solotrain else 'merged/'
    log_path = cnn_inv_log_path(classifier, start_layer, rec_layer) + save_subdir
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    ni = NetInversion(module_list, log_path, classifier=classifier)
    img_paths = subset10_paths(classifier)
    img_mat = load_and_stack_imgs(img_paths)
    to_fetch = retrieve_special or ('{}/{}:0'.format(module_list[-1].name, module_list[-1].rec_name),)
    recs = ni.run_model_on_images(img_mat, to_fetch)
    for name, rec in zip(to_fetch, recs):
        print(rec.shape)
        np.save('{}cnn_rec_{}.npy'.format(log_path, name.replace('/', '_')), rec)

    if retrieve_special is None:
        images = np.split(recs[0], indices_or_sections=recs[0].shape[0], axis=0)
        image_ids = [p.split('/')[-1].split('.')[0][len('val'):] for p in img_paths]
        image_save_paths = [log_path + 'img_rec_{}.png'.format(i) for i in image_ids]
        for path, img in zip(image_save_paths, images):
            skimage.io.imsave(path, img)
