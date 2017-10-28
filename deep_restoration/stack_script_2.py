from utils.db_benchmark import db_lin_to_img_gen


db_lin_to_img_gen('alexnet', use_solotrain=True)
db_lin_to_img_gen('alexnet', use_solotrain=False)
