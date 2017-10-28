from utils.db_benchmark import db_lin_to_img_gen, run_stacked_module


run_stacked_module('alexnet', 2, 1, use_solotrain=True, subdir_name=None)
# db_lin_to_img_gen('alexnet', use_solotrain=True)
# db_lin_to_img_gen('alexnet', use_solotrain=False)
