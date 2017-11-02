from foe_inv_funcs import run_image_opt_inversions

# run_stacked_module('alexnet', 1, 1, use_solotrain=True, subdir_name=None)
# db_lin_to_img_gen('alexnet', use_solotrain=True)
# db_lin_to_img_gen('alexnet', use_solotrain=False)
run_image_opt_inversions('alexnet', 'full512')
