import matplotlib
matplotlib.use('tkagg', force=True)
import matplotlib.pyplot as plt
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import seaborn as sns


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


def plot_opt_inv_experiment(path, exp_subdirs, log_tags, logscale=True, log_subdir='summaries/', max_steps=None):
    sns.set_style('darkgrid')
    sns.set_context('paper')
    exp_logs = dict()
    for exp in exp_subdirs:
        exp_path = os.path.join(path, exp_subdirs[exp], log_subdir)
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
                if max_steps is not None:
                    steps = steps[:max_steps]
                    values = values[:max_steps]
                plt.plot(steps, values, label=exp)

        if logscale:
            plt.yscale('log')
        plt.xlabel('Iterations')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.show()
        plt.close()


