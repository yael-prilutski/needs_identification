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
        self.trial_chunk = int(self.sec_hz * 6 / self.bin_value)

    def load_clusters(self, data_name, reward_type):
        chunk = self.trial_chunk
        cluster_labels = get_cluster_labels(data_name, join(MANIFOLD_PATH, f'{WEEK_TYPE}_{reward_type}'))
        clusters_by_trials = [cluster_labels[i:i + chunk] for i in range(0, len(cluster_labels), chunk)]
        assert len(cluster_labels) % chunk == 0
        return clusters_by_trials

    def visualize(self, reward_type, clusters_dict):
        day = self.days[0]
        if reward_type == 'water':
            titles = ['water_day1', 'water_day2', 'food_day1']
            trials_primary = [day.drank, day.not_drank]
            trials_secondary = [day.ate, day.not_ate, day.not_ate_licked, day.omission_food_taste]
        else:
            titles = ['food_day1', 'food_day2', 'water_day1']
            trials_primary = [day.ate, day.not_ate, day.not_ate_licked, day.omission_food_taste]
            trials_secondary = [day.drank, day.not_drank]

        day1_primary = [clusters_dict[f'day1_{reward_type}'][i]
                        for i, v in enumerate(clusters_dict[f'day1_{reward_type}_relevant_trials'])
                        if clusters_dict[f'day1_{reward_type}_trials_type'][v - 1] in trials_primary]
        day2_primary = [clusters_dict[f'day2_{reward_type}'][i]
                        for i, v in enumerate(clusters_dict[f'day2_{reward_type}_relevant_trials'])
                        if clusters_dict[f'day2_{reward_type}_trials_type'][v - 1] in trials_primary]
        day1_secondary = [clusters_dict[f'day1_{reward_type}'][i]
                          for i, v in enumerate(clusters_dict[f'day1_{reward_type}_relevant_trials'])
                          if clusters_dict[f'day1_{reward_type}_trials_type'][v - 1] in trials_secondary]

        cmap = ListedColormap([COLOR_MAP[i] for i in sorted(COLOR_MAP.keys())])
        fig = plt.figure(figsize=(30, 20))
        plt.suptitle(f'{WEEK_TYPE} {reward_type} trials', fontsize=30)
        for i, trials in enumerate([day1_primary, day2_primary, day1_secondary]):
            plt.subplot(1, 3, i + 1)
            plt.title(titles[i], fontsize=25)
            if len(trials):
                plt.imshow(trials, cmap=cmap, interpolation='nearest', aspect='auto', vmin=0, vmax=7)
                plt.colorbar(label='Value')
        plt.savefig(join(RESULTS_PATH, 'manifold', 'heatmaps', '2_days_clusters',
                         f'{WEEK_TYPE}_{reward_type}_{self.mouse.name}.jpg'))
        plt.close()

    def run(self):
        clusters_dict = {}
        for reward_type in ['water', 'food']:
            data_name = f'{self.mouse.name}_{WEEK_TYPE}_{reward_type}'
            if reward_type == 'water':
                day1 = self.days[0]
                day2 = self.days[1]
            else:
                day1 = self.days[2]
                day2 = self.days[3]
            iti_dict = np.load(
                join(self.mouse.week.data_dir_path, 'axes', f'{self.mouse.name}_dot_products_dict.npy'), allow_pickle=True)[()]
            relevant_trials1 = iti_dict[day1.name]['relevant_trials']
            relevant_trials2 = iti_dict[day2.name]['relevant_trials']

            clusters = self.load_clusters(data_name, reward_type)
            clusters_dict[f'day1_{reward_type}'] = clusters[:len(relevant_trials1)]
            clusters_dict[f'day1_{reward_type}_relevant_trials'] = relevant_trials1
            clusters_dict[f'day1_{reward_type}_trials_type'] = day1.load_data_dict()['trials_classification']
            clusters_dict[f'day2_{reward_type}'] = clusters[len(relevant_trials1):]
            clusters_dict[f'day2_{reward_type}_relevant_trials'] = relevant_trials2
            clusters_dict[f'day2_{reward_type}_trials_type'] = day2.load_data_dict()['trials_classification']
            self.visualize(reward_type, clusters_dict)

        np.save(join(self.week_data, '2_days_ITI_clusters_by_trials.npy'), clusters_dict)


def main():
    for mouse_dict in MICE:
        if DAYS in mouse_dict.keys():
            mouse = Mouse(mouse_dict, mouse_dict[DAYS], mouse_dict[WEEK_TYPE])
            process = ClustersByTrials(mouse)
            process.run()
            print(f'finished mouse {mouse.name}')


'__main__' == __name__ and main()
