import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from os import makedirs
from os.path import join
from scipy.ndimage import uniform_filter1d

from configurations import *
from mouse import Mouse
from Processor import Processor
from axes.axes_analyzer import AxesAnalyzer


RELEVANT_MICE = ALL_MICE
BEFORE_BEHAVIOR = False
SMOOTH = True
WEEK = 'week2'
DAYS = 'week2_days'
DICT_NAME = 'dot_products_dict.npy'
PATH = join(RESULTS_PATH, 'axes', 'all_week_1d')
if not SMOOTH:
    PATH = join(PATH, 'no_smooth')

makedirs(PATH, exist_ok=True)


class MixedDays1dBehavior(Processor):

    def __init__(self, mouse, dict_name, *args, **kwargs):
        Processor.__init__(self, *args, **kwargs)
        self.mouse = mouse
        self.days = mouse.days
        self.week = mouse.week
        self.flat_parameter = int(6 * mouse.smooth_factor)
        self.analyzer = AxesAnalyzer(mouse.smooth_factor)
        self.iti_size = self.analyzer.iti_slice
        self.dict_path = join(self.week.data_dir_path, 'axes', f'{self.mouse.name}_{dict_name}')

    def visualize(self, dot_products, colors, day1_blocks, day3_blocks):
        titles = [f'day{i + 1}' for i in range(4)]
        axes_names = ['Thirst', 'Hunger']

        plt.figure(figsize=[30, 30])
        plt.suptitle(f'All week 1D- {self.mouse.name}', fontsize=30)

        for axis in range(2):
            axis_name = axes_names[axis]
            for day in range(4):
                day_name = titles[day]
                data = dot_products[day][axis]
                color = colors[day]
                min_value = max(min(0, min(data) - 0.1), -0.5)
                max_value = min(max(1, max(data) + 0.1), 1.5)

                plt.subplot(4, 2, (day * 2 + 1) + axis)
                plt.title(f'{day_name} {axis_name}', fontsize=20)
                plt.scatter(np.arange(len(data)), data, c=color, edgecolors='None')
                plt.ylabel(axis_name, fontsize=20)
                plt.ylim(min_value, max_value)
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)

                if day == 0:
                    for frame in day1_blocks[0]:
                        plt.axvline(x=frame, color='blue', linestyle='--', linewidth=1.5)
                    for frame in day1_blocks[1]:
                        plt.axvline(x=frame, color='red', linestyle='--', linewidth=1.5)
                elif day == 2:
                    for frame in day3_blocks[0]:
                        plt.axvline(x=frame, color='blue', linestyle='--', linewidth=1.5)
                    for frame in day3_blocks[1]:
                        plt.axvline(x=frame, color='red', linestyle='--', linewidth=1.5)

        # plt.show()
        plt.savefig(join(PATH, f'{self.mouse.name}_1d_all_week.jpg'))
        plt.close()

    def find_blocks_location(self, day, relevant_trials):
        trials_type = day.load_data_dict()['trials_classification']
        food_trials = []
        water_trials = [relevant_trials[0]]

        current = 'water'
        for i in relevant_trials[1:]:
            if trials_type[i - 1] in [day.ate, day.omission_food_taste, day.not_ate, day.not_ate_licked,
                                      day.omission_pellet]:
                if current == 'food':
                    continue
                else:
                    food_trials.append(i)
                    current = 'food'

            elif trials_type[i - 1] in [day.drank, day.not_drank, day.omission_water]:
                if current == 'water':
                    continue
                else:
                    water_trials.append(i)
                    current = 'water'

        final_food_trials = [int(i * self.iti_size) for i, v in enumerate(relevant_trials) if v in food_trials]
        final_water_trials = [int(i * self.iti_size) for i, v in enumerate(relevant_trials) if v in water_trials]

        return final_water_trials, final_food_trials

    def run(self):
        products_dict = np.load(self.dict_path, allow_pickle=True)[()]

        dot_products = []
        colors_classification = []
        for i, day in enumerate(self.days[:4]):
            last_trial = products_dict[day.name]['last_vector_trial']
            length_day = np.where(products_dict[day.name]['relevant_trials'] == last_trial)[0][0] * self.iti_size
            axis_thirst = products_dict[day.name]['water_ortho'][:length_day]
            axis_hunger = products_dict[day.name]['food_ortho'][:length_day]
            day_colors = self.analyzer.classify_by_behavior_colors(
                day, products_dict[day.name], length_day, self.iti_size, before_behavior=BEFORE_BEHAVIOR)
            if SMOOTH:
                axis_thirst = uniform_filter1d(axis_thirst, self.flat_parameter)
                axis_hunger = uniform_filter1d(axis_hunger, self.flat_parameter)
            dot_products.append([axis_thirst, axis_hunger])
            colors_classification.append(day_colors)

        day1_blocks = self.find_blocks_location(self.days[0], products_dict[self.days[0].name]['relevant_trials'])
        day3_blocks = self.find_blocks_location(self.days[2], products_dict[self.days[2].name]['relevant_trials'])

        self.visualize(dot_products, colors_classification, day1_blocks, day3_blocks)


def main():
    for mouse_p in RELEVANT_MICE:
        if WEEK in mouse_p.keys():
            mouse = Mouse(mouse_p, mouse_p[DAYS], mouse_p[WEEK])
            process = MixedDays1dBehavior(mouse, dict_name=DICT_NAME)
            process.run()
            print(f'finished processing {mouse.name}')


'__main__' == __name__ and main()
