from foe_inv_funcs import run_image_opt_inversions
import numpy as np
from skimage.io import imsave


def run_fc(select_img):
    run_image_opt_inversions('vgg16', 'full512', layer_select='fc7/lin', lr=.3,
                             mse_iterations=10000, opt_iterations=15000,
                             jitterations=6900,
                             select_img=select_img)

    run_image_opt_inversions('vgg16', 'full512', layer_select='fc8/lin', lr=.3,
                             mse_iterations=10000, opt_iterations=15000,
                             jitterations=6900,
                             select_img=select_img)


def mv_fig12(n_iterations=8):
    layers = ['conv5_2/lin', 'fc7/lin']

    img_hw = 224
    subset_dir = '../data/selected/images_resized_{}/'.format(img_hw)
    img_names = ['val13_monkey', 'val43_flamingo', 'stock_abstract']
    tgt_paths = ['{}{}.bmp'.format(subset_dir, i) for i in img_names]

    for it in range(n_iterations):
        img_subdirs = ['{}_{}'.format(k, it) for k in img_names]

        for layer in layers:
            run_image_opt_inversions('vgg16', 'full512', layer_select=layer, lr=.3,
                                     mse_iterations=10000, opt_iterations=15000,
                                     jitterations=6900,
                                     select_img=None, custom_images=(tgt_paths, img_subdirs))


def acc_modes(img_ids=(53, 99, 160), iterations=4, rec_it=17000):
    base_path = '/home/frederik/Desktop/Thesis/mode_exploration/'
    for layer in ('fc7', 'fc8'):
        imgs_list = []
        for img_id in img_ids:
            it_list = []
            for it in range(iterations):
                path = base_path + 'it{}/{}/val{}/full512/mats/rec_{}.npy'.format(it, layer, img_id, rec_it)
                it_list.append(np.load(path)[0, ...])
            imgs_list.append(it_list)
            if iterations == 4:
                collage = np.concatenate((np.concatenate(it_list[:2], axis=1), np.concatenate(it_list[2:], axis=1)), axis=0)

                p1 = np.percentile(collage, 1)
                p99 = np.percentile(collage, 99)
                collage = np.minimum(np.maximum(collage, p1), p99)
                collage = (collage - np.min(collage)) / (np.max(collage) - np.min(collage))

                imsave(base_path + '{}_img{}_rec{}.png'.format(layer, img_id, rec_it), collage)

        collage = np.concatenate([np.concatenate(k, axis=1) for k in imgs_list], axis=0)

        p1 = np.percentile(collage, 1)
        p99 = np.percentile(collage, 99)
        collage = np.minimum(np.maximum(collage, p1), p99)
        collage = (collage - np.min(collage)) / (np.max(collage) - np.min(collage))

        imsave(base_path + '{}_rec{}.png'.format(layer, rec_it), collage)


mv_fig12()
# acc_modes()
