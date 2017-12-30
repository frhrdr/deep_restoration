from foe_inv_funcs import run_image_opt_inversions

# run_stacked_module('alexnet', 1, 1, use_solotrain=True, subdir_name=None)
# db_lin_to_img_gen('alexnet', use_solotrain=True)
# db_lin_to_img_gen('alexnet', use_solotrain=False)

run_image_opt_inversions('alexnet', 'full512', layer_select='lin', lr=3., mse_iterations=5000, opt_iterations=10000,
                         select_img=None)

# run_image_opt_inversions('vgg16', 'full512', layer_select='fc7/lin', lr=1., mse_iterations=5000, opt_iterations=10000,
#                          select_img=6)
#
# run_image_opt_inversions('vgg16', 'full512', layer_select='fc8/lin', lr=1., mse_iterations=5000, opt_iterations=10000,
#                          select_img=6)
