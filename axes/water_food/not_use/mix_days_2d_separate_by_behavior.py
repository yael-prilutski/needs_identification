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
WEEK = 'week2'
DAYS = 'week2_days'
DICT_NAME = 'dot_products_dict.npy'
PATH = join(RESULTS_PATH, 'axes', 'mixed_2d_separate_by_behavior')
SMOOTH = True
if not SMOOTH:
    PATH = join(PATH, 'no_smooth')
makedirs(PATH, exist_ok=True)


class BetweenDaysAxis(Processor):

    def __init__(self, mouse, dict_name, *args, **kwargs):
        Processor.__init__(self, *args, **kwargs)
        self.mouse = mouse
        self.days = mouse.days
        self.week = mouse.week
        self.flat_parameter = int(5 * 3 * mouse.smooth_factor)
        self.iti_size = AxesAnalyzer(mouse.smooth_factor).iti_slice
        self.dict_path = join(self.week.data_dir_path, 'axes', f'{self.mouse.name}_{dict_name}')

    def visualize(self, dot_products, colors):
        titles = ['day1- water', 'day1- food', 'day3- water', 'day3- food']

        plt.figure(figsize=[27, 21])
        plt.suptitle(f'Mix days 2d by behavior neutral trials- {self.mouse.name}', fontsize=30)

        for i_day in range(4):
            thirst, hunger = dot_products[i_day]
            min_value = min(0, min(thirst) - 0.1, min(hunger) - 0.1)
            max_value = max(1, max(thirst) + 0.1, max(hunger) + 0.1)

            plt.subplot(2, 2, i_day + 1)
            plt.title(titles[i_day], fontsize=20)
            plt.scatter(thirst, hunger, c=colors[i_day], edgecolors='None', alpha=0.5)
            plt.ylabel('Hunger', fontsize=20)
            plt.xlabel('Thirst', fontsize=20)
            plt.xlim(min_value, max_value)
            plt.ylim(min_value, max_value)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.gca().set_aspect('equal', adjustable='box')

        # plt.show()
        plt.savefig(join(PATH, f'{self.mouse.name}_mix_days_2d_by_behavior.jpg'))
        plt.close()

    def classify_colors(self, day, day_data):
        last_trial = day_data['last_vector_trial']
        relevant_trials = [i for i in day_data['relevant_trials'] if i < last_trial]
        trials_type = day_data['trials_type']
        colors_water = []
        colors_food = []
        for i in relevant_trials[1:]:
            trial = trials_type[i - 1]
            if trial == day.drank:
                colors_water.extend(['darkblue'] * self.iti_size)
            elif trial == day.not_drank:
                colors_water.extend(['lightblue'] * self.iti_size)
            elif trial == day.omission_water:
                colors_water.extend(['green'] * self.iti_size)
            elif trial in [day.ate, day.omission_food_taste]:
                colors_food.extend(['darkred'] * self.iti_size)
            elif trial in [day.not_ate, day.not_ate_licked]:
                colors_food.extend(['lightcoral'] * self.iti_size)
            elif trial == day.omission_pellet:
                colors_food.extend(['purple'] * self.iti_size)
        return colors_water, colors_food

    def activity_by_behavior(self, thirst_axis, hunger_axis, day, day_data):
        last_trial = day_data['last_vector_trial']
        relevant_trials = [i for i in day_data['relevant_trials'] if i < last_trial]
        trials_type = day_data['trials_type']
        thirst_water = []
        thirst_food = []
        hunger_water = []
        hunger_food = []
        for i, v in enumerate(relevant_trials[1:]):
            trial = trials_type[v - 1]
            if trial in [day.drank, day.not_drank, day.omission_water]:
                thirst_water.extend(thirst_axis[i * self.iti_size: (i + 1) * self.iti_size])
                hunger_water.extend(hunger_axis[i * self.iti_size: (i + 1) * self.iti_size])
            elif trial in [day.ate, day.omission_food_taste, day.not_ate, day.not_ate_licked, day.omission_pellet]:
                thirst_food.extend(thirst_axis[i * self.iti_size: (i + 1) * self.iti_size])
                hunger_food.extend(hunger_axis[i * self.iti_size: (i + 1) * self.iti_size])
        return thirst_water, thirst_food, hunger_water, hunger_food

    def run(self):
        products_dict = np.load(self.dict_path, allow_pickle=True)[()]
        relevant_days = [self.days[0], self.days[2]]

        dot_products = []
        colors_classification = []
        for i, day in enumerate(relevant_days):
            last_trial = products_dict[day.name]['last_vector_trial']
            length_day = np.where(products_dict[day.name]['relevant_trials'] == last_trial)[0][0] * self.iti_size
            axis_thirst = products_dict[day.name]['water_ortho'][:length_day]
            axis_hunger = products_dict[day.name]['food_ortho'][:length_day]
            colors_water, colors_food = self.classify_colors(day, products_dict[day.name])
            thirst_water, thirst_food, hunger_water, hunger_food = self.activity_by_behavior(
                axis_thirst, axis_hunger, day, products_dict[day.name])
            if SMOOTH:
                thirst_water, thirst_food, hunger_water, hunger_food = [
                    uniform_filter1d(trials, size=self.flat_parameter)
                    for trials in [thirst_water, thirst_food, hunger_water, hunger_food]]
            dot_products.extend([[thirst_water, hunger_water], [thirst_food, hunger_food]])
            colors_classification.extend([colors_water, colors_food])

        self.visualize(dot_products, colors_classification)
        return dot_products


def main():
    for mouse_p in RELEVANT_MICE:
        if WEEK in mouse_p.keys():
            mouse = Mouse(mouse_p, mouse_p[DAYS], mouse_p[WEEK])
            process = BetweenDaysAxis(mouse, dict_name=DICT_NAME)
            process.run()
            print(f'finished processing {mouse.name}')


'__main__' == __name__ and main()
