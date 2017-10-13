import matplotlib
matplotlib.use('tkagg', force=True)
import matplotlib.pyplot as plt
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def prepare_scalar_logs(path):
    size_guidance = {'compressedHistograms': 1, 'images': 1, 'audio': 1, 'scalars': 0, 'histograms': 1, 'tensors': 1}
    event_acc = EventAccumulator(path, size_guidance=size_guidance)
    event_acc.Reload()
    scalar_logs = dict()
    for tag in event_acc.Tags()['scalars']:
        events = event_acc.Scalars(tag)
        steps = [k.step for k in events]
        values = [k.value for k in events]
        scalar_logs[tag] = (steps, values)
    return scalar_logs


def plot_opt_inv_experiment(path, exp_subdirs, log_tags):
    exp_logs = dict()
    for exp in exp_subdirs:
        exp_path = os.path.join(path, exp_subdirs[exp], 'summaries')
        print(exp_path)
        exp_logs[exp] = prepare_scalar_logs(exp_path)

    for log_name in log_tags:
        tag = log_tags[log_name]
        plt.figure()
        plt.title(log_name)
        for exp in exp_logs:
            log = exp_logs[exp]
            print(log, tag, exp_logs)
            if tag in log:
                steps, values = log[tag]
                plt.plot(steps, values, label=exp)
        plt.legend()
        plt.show()
        plt.close()


def plot_example_exp():
    path = '../logs/opt_inversion/alexnet/slim_vs_img/c2l_to_c1l'
    exp_subdirs = {'No prior': 'no_prior',
                   'Pre-image with prior': 'pre_image_8x8_full_prior/1e-3',
                   'Pre-image with no prior': 'pre_image_no_prior'}
    log_tags = {'Total loss': 'Total_Loss',
                'Reconstruction error': 'MSE_Reconstruction_1'}
    plot_opt_inv_experiment(path, exp_subdirs, log_tags)
