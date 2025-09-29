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
PATH = join(RESULTS_PATH, 'manifold', 'heatmaps', 'clusters_by_blocks')


class ClustersByTrials:

    def __init__(self, mouse):
        self.base_path = BASE_PATH
        self.mouse = mouse
        self.days = self.mouse.days
        self.week_data = mouse.week.data_dir_path
        self.sec_hz = mouse.days[0].smooth_factor
        self.bin_value = int(self.sec_hz / 6)

    def process_day(self, clusters, neutral_trials, day_i):
        day = self.days[day_i]
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

        water_block = [clusters[i] for i, v in enumerate(neutral_trials) if v in water_trials]
        food_block = [clusters[i] for i, v in enumerate(neutral_trials) if v in food_trials]

        return water_block, food_block

    def visualize(self, day1, day3):
        cmap = ListedColormap([COLOR_MAP[i] for i in sorted(COLOR_MAP.keys())])
        titles = ['day1 water', 'day1 food', 'day3 water', 'day3 food']

        fig = plt.figure(figsize=(30, 30))
        plt.suptitle(f'{WEEK_TYPE} blocks clusters', fontsize=30)

        for i, plot in enumerate([*day1, *day3]):
            plt.subplot(2, 2, i + 1)
            plt.title(titles[i], fontsize=25)
            if len(plot):
                plt.imshow(plot, cmap=cmap, interpolation='nearest', aspect='auto', vmin=0, vmax=7)
                plt.colorbar(label='Value')
        plt.savefig(join(PATH, f'{self.mouse.name}_blocks_clusters.jpg'))
        plt.close()

    def run(self):
        clusters_dict = np.load(join(self.week_data, 'ITI_clusters_by_trials.npy'), allow_pickle=True)[()]
        day1_trials = self.process_day(clusters_dict['day1'], clusters_dict['day1_relevant_trials'], 0)
        day3_trials = self.process_day(clusters_dict['day3'], clusters_dict['day3_relevant_trials'], 2)
        self.visualize(day1_trials, day3_trials)


def main():
    for mouse_dict in MICE:
        if DAYS in mouse_dict.keys():
            mouse = Mouse(mouse_dict, mouse_dict[DAYS], mouse_dict[WEEK_TYPE])
            process = ClustersByTrials(mouse)
            process.run()
            print(f'finished mouse {mouse.name}')


'__main__' == __name__ and main()
