import numpy as np
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams['svg.fonttype'] = 'none'

from configurations import *
from mouse import Mouse
from manifold.visualize_datasets import COLOR_MAP, get_cluster_labels

MICE = ALL_MICE
WEEK_TYPE = 'week2'
DAYS = f'{WEEK_TYPE}_days'


class ClustersByTrials:

    def __init__(self, mouse):
        self.base_path = BASE_PATH
        self.mouse = mouse
        self.days = self.mouse.days
        self.week_data = mouse.week.data_dir_path
        self.sec_hz = mouse.days[0].smooth_factor
        self.bin_value = int(self.sec_hz / 6)

    def load_clusters(self, data_name, day_i):
        chunk = int(self.sec_hz * 6 / self.bin_value)
        if WEEK_TYPE == 'week1':
            cluster_labels = get_cluster_labels(data_name, join(MANIFOLD_PATH, f'day{day_i + 1}'))
        else:
            cluster_labels = get_cluster_labels(data_name, join(MANIFOLD_PATH, f'week2_day{day_i + 1}'))
        clusters_by_trials = [cluster_labels[i:i + chunk] for i in range(0, len(cluster_labels), chunk)]
        assert len(cluster_labels) % chunk == 0
        return clusters_by_trials

    def visualize(self, clusters, relevant_trials, day_i, day):
        trials_type = day.load_data_dict()['trials_classification']

        titles = ['drank', 'not_drank', 'ate', 'not_ate', 'neutral']
        drank = [clusters[i] for i, v in enumerate(relevant_trials) if trials_type[v - 1] == day.drank]
        not_drank = [clusters[i] for i, v in enumerate(relevant_trials) if trials_type[v - 1] == day.not_drank]
        ate = [clusters[i] for i, v in enumerate(relevant_trials) if trials_type[v - 1] in [day.ate, day.omission_food_taste]]
        not_ate = [clusters[i] for i, v in enumerate(relevant_trials) if trials_type[v - 1] in [day.not_ate, day.not_ate_licked]]
        neutral = [clusters[i] for i, v in enumerate(relevant_trials) if trials_type[v - 1] == day.neutral]

        cmap = ListedColormap([COLOR_MAP[i] for i in sorted(COLOR_MAP.keys())])
        fig = plt.figure(figsize=(30, 30))
        plt.suptitle(f'{WEEK_TYPE} Day{day_i} trials', fontsize=30)
        for i, plot in enumerate([drank, not_drank, ate, not_ate, neutral]):
            plt.subplot(3, 2, i + 1)
            plt.title(titles[i], fontsize=25)
            if len(plot):
                plt.imshow(plot, cmap=cmap, interpolation='nearest', aspect='auto', vmin=0, vmax=7)
                plt.colorbar(label='Value')
        plt.savefig(join(RESULTS_PATH, 'manifold', 'heatmaps', f'{WEEK_TYPE}_heatmaps_after_behavior',
                         f'{WEEK_TYPE}_{self.mouse.name}_day{day_i}.jpg'))
        plt.close()

    def run(self):
        clusters_dict = {}
        for day_i in [0, 2]:
            if WEEK_TYPE == 'week1':
                data_name = f'{self.mouse.name}_day{day_i + 1}'
            else:
                data_name = f'{self.mouse.name}_week2_day{day_i + 1}'
            day = self.days[day_i]
            iti_dict = np.load(
                join(self.mouse.week.data_dir_path, 'axes', f'{self.mouse.name}_dot_products_dict.npy'), allow_pickle=True)[()]
            relevant_trials = iti_dict[day.name]['relevant_trials']
            clusters = self.load_clusters(data_name, day_i)
            clusters_dict[f'day{day_i + 1}'] = clusters
            clusters_dict[f'day{day_i + 1}_relevant_trials'] = relevant_trials
            self.visualize(clusters, relevant_trials, day_i + 1, day)

        np.save(join(self.week_data, 'ITI_clusters_by_trials.npy'), clusters_dict)


def main():
    for mouse_dict in MICE:
        if DAYS in mouse_dict.keys():
            mouse = Mouse(mouse_dict, mouse_dict[DAYS], mouse_dict[WEEK_TYPE])
            process = ClustersByTrials(mouse)
            process.run()
            print(f'finished mouse {mouse.name}')


'__main__' == __name__ and main()
