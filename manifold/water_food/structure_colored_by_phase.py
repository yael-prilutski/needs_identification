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
PATH = join(RESULTS_PATH, 'manifold', 'structure_phases')
if WEEK_TYPE == 'week1':
    PATH = join(PATH, WEEK_TYPE)
WATER_COLOR = '#3f9bcc'
FOOD_COLOR = '#c91912'


class ClustersByTrials:

    def __init__(self, mouse):
        self.mouse = mouse
        self.week_data = mouse.week.data_dir_path
        bin_value = 6
        self.trial_size = bin_value * 6

    def label_data(self, day_i, neutral_trials):
        day = self.mouse.days[day_i]
        trials_type = day.load_data_dict()['trials_classification']
        food_trials = []
        water_trials = [neutral_trials[0]]

        current = 'water'
        for i in neutral_trials[1:]:
            if trials_type[i - 1] in [day.ate, day.omission_food_taste, day.not_ate, day.not_ate_licked,
                                      day.omission_pellet]:
                food_trials.append(i)
                current = 'food'

            elif trials_type[i - 1] in [day.drank, day.not_drank, day.omission_water]:
                water_trials.append(i)
                current = 'water'
            else:
                if current == 'water':
                    water_trials.append(i)
                else:
                    food_trials.append(i)

        labels = []
        for i in neutral_trials:
            if i in water_trials:
                labels.extend([1] * self.trial_size)
            elif i in food_trials:
                labels.extend([2] * self.trial_size)
            else:
                labels.extend([3] * self.trial_size)

        return labels

    def process_day(self, neutral_trials, day_i):
        if WEEK_TYPE == 'week1':
            data_name = f'{self.mouse.name}_day{day_i}'
            internal._set_base_directory(join(MANIFOLD_PATH, f'day{day_i}'))
        else:
            data_name = f'{self.mouse.name}_week2_day{day_i}'
            internal._set_base_directory(join(MANIFOLD_PATH, f'week2_day{day_i}'))
        dataset = internal.get_dataset(data_name, alias="lem_final")
        labels = self.label_data(day_i - 1, neutral_trials)
        color_map = {1: WATER_COLOR, 2: FOOD_COLOR, 3: '#0d0c0c'}

        visualize_dataset(dataset,
                          join(PATH, f'{self.mouse.name}_day{day_i}.jpg'),
                          default=False,
                          variable=labels,
                          color_map=color_map,
                          alpha=0.05)

    def run(self):
        clusters_dict = np.load(join(self.week_data, 'ITI_clusters_by_trials.npy'), allow_pickle=True)[()]
        self.process_day(clusters_dict['day1_relevant_trials'], 1)
        self.process_day(clusters_dict['day3_relevant_trials'], 3)


def main():
    for mouse_dict in MICE:
        if DAYS in mouse_dict.keys():
            mouse = Mouse(mouse_dict, mouse_dict[DAYS], mouse_dict[WEEK_TYPE])
            process = ClustersByTrials(mouse)
            process.run()
            print(f'finished mouse {mouse.name}')


'__main__' == __name__ and main()
