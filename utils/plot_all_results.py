import os
import json
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict


def plot_all(path, res_name):
    """
    Args:
        path - path to folder with HistorySaver's results
        res_name - plot title and file name

    Given a HistorySaver' logs (containing folders with results of several experiments) plots all results in one plot.
    Saves the plot in the 'path' directory.
    """

    subdirectories = [dI for dI in os.listdir(path) if os.path.isdir(os.path.join(path, dI))]
    results = defaultdict(dict)
    for subdirectory in subdirectories:
        for results_names in ['loss_history', 'src_metrics', 'trg_metrics']:
            current_path = os.path.join(path, subdirectory, results_names)
            if os.path.exists(current_path):
                with open(current_path, 'r') as f:
                    results[results_names][subdirectory] = json.load(f)
    plt.figure(figsize=(22, 8))

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan']

    linestyles = ['-', '--', '-.', ':']

    for i, results_names in enumerate(['loss_history', 'src_metrics', 'trg_metrics']):
        plt.subplot(1, 3, i + 1)
        for j, experiment in enumerate(sorted(results[results_names])):
            for k, metric in enumerate(sorted(results[results_names][experiment])):
                plt.plot(results[results_names][experiment][metric], label=str(experiment + " " + metric),
                         color=colors[j % len(results[results_names])],
                         linestyle=linestyles[k % len(results[results_names][experiment])])

                if results_names == 'trg_metrics':
                    print('{} {}\t{:0.5f}'.format(experiment, metric, max(results[results_names][experiment][metric])))
        plt.grid()
        plt.legend()
        plt.title('{} for all experiments'.format(results_names))

    plt.savefig(os.path.join(path, res_name))


if __name__ == '__main__':
    plot_all('./_log/0430_all_frozen_141', 'DANN_with_resnet_frozen_141')
