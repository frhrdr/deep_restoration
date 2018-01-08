from foe_inv_funcs import run_image_opt_inversions


def run_fc(select_img):
    run_image_opt_inversions('vgg16', 'full512', layer_select='fc6/lin', lr=.3, mse_iterations=10000, opt_iterations=15000,
                             jitterations=6900,
                             select_img=select_img)

    run_image_opt_inversions('vgg16', 'full512', layer_select='fc7/lin', lr=.3, mse_iterations=10000, opt_iterations=15000,
                             jitterations=6900,
                             select_img=select_img)

    run_image_opt_inversions('vgg16', 'full512', layer_select='fc8/lin', lr=.3, mse_iterations=10000, opt_iterations=15000,
                             jitterations=6900,
                             select_img=select_img)


run_fc(0)
run_fc(3)
run_fc(9)