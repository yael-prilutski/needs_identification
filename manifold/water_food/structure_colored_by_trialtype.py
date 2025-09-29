import numpy as np
from os.path import join
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from manifold.visualize_datasets import visualize_dataset
from manifold import internal

from configurations import *
from mouse import Mouse

MICE = ALL_MICE
WEEK_TYPE = 'week1'
DAYS = f'{WEEK_TYPE}_days'
PATH = join(RESULTS_PATH, 'manifold', 'structure_trial_type', WEEK_TYPE)
DRANK_COLOR = '#3f9bcc'
NOT_DRANK_COLOR = '#1a23d6'
ATE_COLOR = '#f51f16'
NOT_ATE_COLOR = '#f4b6b0'
OMISSION_WATER = '#0ced57'
OMISSION_FOOD = '#e434f7'
NEUTRAL = '#0d0c0c'


class ClustersByTrials:

    def __init__(self, mouse):
        self.mouse = mouse
        self.week_data = mouse.week.data_dir_path
        bin_value = 6
        self.trial_size = bin_value * 6

    def label_data(self, day_i, neutral_trials):
        day = self.mouse.days[day_i]
        trials_type = day.load_data_dict()['trials_classification']

        labels = []
        for i in neutral_trials:
            if trials_type[i - 1] in [day.ate, day.omission_food_taste]:
                labels.extend([3] * self.trial_size)
            elif trials_type[i - 1] in [day.not_ate, day.not_ate_licked]:
                labels.extend([4] * self.trial_size)
            elif trials_type[i - 1] == day.omission_pellet:
                labels.extend([6] * self.trial_size)

            elif trials_type[i - 1] == day.drank:
                labels.extend([1] * self.trial_size)
            elif trials_type[i - 1] == day.not_drank:
                labels.extend([2] * self.trial_size)
            elif trials_type[i - 1] == day.omission_water:
                labels.extend([5] * self.trial_size)

            else:
                if len(labels) == 0:
                    labels.extend([7] * self.trial_size)
                else:
                    labels.extend([labels[-1]] * self.trial_size)

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
        color_map = {1: DRANK_COLOR, 2: NOT_DRANK_COLOR, 3: ATE_COLOR,
                     4: NOT_ATE_COLOR, 5: OMISSION_WATER, 6: OMISSION_FOOD, 7: NEUTRAL}

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
