import numpy as np
from os.path import join
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from manifold.visualize_datasets import visualize_dataset
from manifold import internal

from configurations import *
from mouse import Mouse

MICE = ALL_MICE
WEEK_TYPE = 'week2'
DAYS = f'{WEEK_TYPE}_days'
PATH = join(RESULTS_PATH, 'manifold', 'structure_by_days', WEEK_TYPE)
DAY1_COLOR = '#3fcc48'
DAY2_COLOR = '#c91912'


class ClustersByTrials:

    def __init__(self, mouse):
        self.mouse = mouse
        self.week_data = mouse.week.data_dir_path
        bin_value = 6
        self.trial_size = bin_value * 6

    def process(self, dict_iti, reward_type):
        data_name = f'{self.mouse.name}_{WEEK_TYPE}_{reward_type}'
        internal._set_base_directory(join(MANIFOLD_PATH, f'{WEEK_TYPE}_{reward_type }'))
        dataset = internal.get_dataset(data_name, alias="lem_final")
        len_day1 = len(dict_iti[f'day1_{reward_type}_relevant_trials'])
        len_day2 = len(dict_iti[f'day2_{reward_type}_relevant_trials'])
        colors = [1] * len_day1 * self.trial_size + [2] * len_day2 * self.trial_size
        color_map = {1: DAY1_COLOR, 2: DAY2_COLOR}

        visualize_dataset(dataset,
                          join(PATH, f'{self.mouse.name}_{reward_type}.jpg'),
                          default=False,
                          variable=colors,
                          color_map=color_map,
                          alpha=0.05)

    def run(self):
        clusters_dict = np.load(join(self.week_data, '2_days_ITI_clusters_by_trials.npy'), allow_pickle=True)[()]
        for reward in ['water', 'food']:
            self.process(clusters_dict, reward)


def main():
    for mouse_dict in MICE:
        if DAYS in mouse_dict.keys():
            mouse = Mouse(mouse_dict, mouse_dict[DAYS], mouse_dict[WEEK_TYPE])
            process = ClustersByTrials(mouse)
            process.run()
            print(f'finished mouse {mouse.name}')


'__main__' == __name__ and main()
