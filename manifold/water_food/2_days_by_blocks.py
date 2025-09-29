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
PATH = join(RESULTS_PATH, 'manifold', 'structure_by_days_blocks', WEEK_TYPE)
WATER_COLOR = '#3f9bcc'
FOOD_COLOR = '#c91912'


class ClustersByTrials:

    def __init__(self, mouse):
        self.mouse = mouse
        self.week_data = mouse.week.data_dir_path
        bin_value = 6
        self.trial_size = bin_value * 6

    def process_day(self, relevant_trials, trials_type):
        day = self.mouse.days[0]
        color_blocks = [[1] * self.trial_size][0]

        current = 'water'
        for i in relevant_trials[1:]:
            if trials_type[i - 1] in [day.ate, day.omission_food_taste, day.not_ate, day.not_ate_licked,
                                      day.omission_pellet]:
                color_blocks.extend([2] * self.trial_size)
                current = 'food'

            elif trials_type[i - 1] in [day.drank, day.not_drank, day.omission_water]:
                color_blocks.extend([1] * self.trial_size)
                current = 'water'
            else:
                if current == 'water':
                    color_blocks.extend([1] * self.trial_size)
                else:
                    color_blocks.extend([2] * self.trial_size)

        return color_blocks

    def process(self, dict_iti, reward_type):
        data_name = f'{self.mouse.name}_{WEEK_TYPE}_{reward_type}'
        internal._set_base_directory(join(MANIFOLD_PATH, f'{WEEK_TYPE}_{reward_type }'))
        dataset = internal.get_dataset(data_name, alias="lem_final")
        day1_labels = self.process_day(dict_iti[f'day1_{reward_type}_relevant_trials'],
                                       dict_iti[f'day1_{reward_type}_trials_type'])
        len_day2 = len(dict_iti[f'day2_{reward_type}_relevant_trials'])
        if reward_type == 'water':
            colors = day1_labels + [1] * len_day2 * self.trial_size
        else:
            colors = day1_labels + [2] * len_day2 * self.trial_size
        color_map = {1: WATER_COLOR, 2: FOOD_COLOR}

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
