import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
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


log_dir = '../logs/opt_inversion/alexnet/slim_vs_img/c2l_to_c1l/no_prior/summaries/'
# 'events.out.tfevents.1506963765.frederik-ThinkPad-E570'
# plot_tensorflow_log(log_file)
slogs = prepare_scalar_logs(log_dir)
for key in slogs:
    print(key)
    print(slogs[key])
