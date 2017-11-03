# from net_inversion import NetInversion
# from modules.inv_default_modules import get_stacked_module
# from shutil import copyfile
# import os
#
# classifier = 'alexnet'
# start_layer = 7
# rec_layer = 5
# module_list = get_stacked_module(classifier=classifier, alt_load_subdir='alexnet_stack_{}_to_{}'.format(start_layer,
#                                                                                                         rec_layer),
#                                  start_layer=start_layer, rec_layer=rec_layer)
# log_path = '../logs/cnn_inversion/{}/stack_{}_to_{}/'.format(classifier, start_layer, rec_layer)
#
# if not os.path.exists(log_path):
#     os.makedirs(log_path)
# copyfile('./stack_script_1.py', log_path + 'script.py')
#
# modules = module_list + [module_list[-1].get_mse_loss()]
# ni = NetInversion(modules, log_path, classifier=classifier, summary_freq=10, print_freq=10, log_freq=500)
#
# ni.train_on_dataset(n_iterations=3000, batch_size=32, test_set_size=200, test_freq=100,
#                     optim_name='adam', lr_lower_points=((0, 3e-6), (3000, 1e-5), (2000, 3e-6)))

# from utils.rec_evaluation import subset10_paths, selected_img_ids
# from utils.filehandling import img_wall, load_image
# from utils.mv_benchmark import mv_collect_rec_images


# for i in selected_img_ids():
#     img_name = 'val{}'.format(i)
#     mat = mv_collect_rec_images('alexnet', select_images=[img_name])
#     print(mat.shape)
#     mat = np.squeeze(mat, axis=1)
#     img_wall(mat, '{}_img.png'.format(img_name))

# for i in selected_img_ids():
#     mat = db_collect_rec_images('alexnet', select_modules=[1, 4, 7, 8, 9], select_images=[i], merged=False)
#     mat = np.squeeze(mat, axis=1)
#     img_wall(mat, '{}_img.png'.format(i), rows=1)

# mat = db_collect_rec_images('alexnet', merged=True)
# print(mat.shape)
# np.save('db_lin_mat.npy', mat)
# del mat
#
# selected_layers = ['conv{}_lin'.format(i) for i in range(1, 6)] + ['fc{}_lin'.format(i) for i in range(6, 9)]
# mat = mv_collect_rec_images('alexnet', select_layers=selected_layers)
# print(mat.shape)
# np.save('mv_lin_mat.npy', mat)

#
# tgt_imgs = [load_image(i) for i in subset10_paths('alexnet')]
# tgt_imgs = np.asarray(tgt_imgs)
# print(tgt_imgs.shape)
# print(tgt_imgs.max())
# np.save('../logs/rec_comparisons/tgt_imgs.npy', tgt_imgs)
from utils.mean_filter_benchmark import fgsm_mean_filter_exp

fgsm_mean_filter_exp()
