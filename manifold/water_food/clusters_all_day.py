import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams['svg.fonttype'] = 'none'

from configurations import *
from mouse import Mouse
from manifold.visualize_datasets import COLOR_MAP

MICE = ALL_MICE
WEEK_TYPE = 'week2'
DAYS = f'{WEEK_TYPE}_days'


class ClustersByTrials:

    def __init__(self, mouse):
        self.mouse = mouse
        self.week_data = mouse.week.data_dir_path

    def visualize(self, clusters):
        titles = ['day1', 'day3']
        cmap = ListedColormap([COLOR_MAP[i] for i in sorted(COLOR_MAP.keys())])
        fig = plt.figure(figsize=(30, 18))
        plt.suptitle(f'{WEEK_TYPE} full days trials trials', fontsize=30)
        for i in range(2):
            plt.subplot(1, 2, i + 1)
            plt.title(titles[i], fontsize=25)
            plt.imshow(clusters[titles[i]], cmap=cmap, interpolation='nearest', aspect='auto', vmin=0, vmax=7)
            plt.colorbar(label='Value')
        plt.savefig(join(RESULTS_PATH, 'manifold', 'heatmaps', WEEK_TYPE, f'{WEEK_TYPE}_{self.mouse.name}.jpg'))
        plt.close()

    def run(self):
        clusters_dict = np.load(join(self.week_data, 'ITI_clusters_by_trials.npy'), allow_pickle=True)[()]
        self.visualize(clusters_dict)


def main():
    for mouse_dict in MICE:
        if DAYS in mouse_dict.keys():
            mouse = Mouse(mouse_dict, mouse_dict[DAYS], mouse_dict[WEEK_TYPE])
            process = ClustersByTrials(mouse)
            process.run()
            print(f'finished mouse {mouse.name}')


'__main__' == __name__ and main()
