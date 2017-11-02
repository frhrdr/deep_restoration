from foe_inv_funcs import featmap_inv

match_layer = 2
target_layer = 1
jitter_t = 0
weighting = '1e-4'
make_mse = True
restart_adam = False
image_name = 'val153'
pre_image = False
do_plot = True


prior_id = 'fullc1l6000'
custom_target = None

# prior_id = 'full512'
# custom_target = 'pre_featmap/read:0'
featmap_inv(match_layer, target_layer, image_name, prior_id, prior_weighting=weighting, make_mse=make_mse,
            restart_adam=restart_adam, pre_image=pre_image, do_plot=do_plot,
            jitter_t=0, jitter_stop_point=3200, lr=1., bound_plots=True, custom_target=custom_target)

make_mse = False
featmap_inv(match_layer, target_layer, image_name, prior_id, prior_weighting=weighting, make_mse=make_mse,
            restart_adam=restart_adam, pre_image=pre_image, do_plot=do_plot,
            jitter_t=0, jitter_stop_point=3200, lr=1., bound_plots=True)
