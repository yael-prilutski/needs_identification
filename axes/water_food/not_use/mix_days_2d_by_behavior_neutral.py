import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from os import makedirs
from os.path import join
from scipy.ndimage import uniform_filter1d
plt.rcParams['svg.fonttype'] = 'none'

from configurations import *
from mouse import Mouse
from Processor import Processor
from axes.axes_analyzer import AxesAnalyzer

RELEVANT_MICE = ALL_MICE
WEEK = 'week2'
DAYS = 'week2_days'
DICT_NAME = 'dot_products_dict.npy'
PATH = join(RESULTS_PATH, 'axes', 'mixed_2d_by_behavior')
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
        self.analyzer = AxesAnalyzer(mouse.smooth_factor)
        self.iti_size = self.analyzer.iti_slice
        self.dict_path = join(self.week.data_dir_path, 'axes', f'{self.mouse.name}_{dict_name}')

    def visualize(self, dot_products, colors):
        good_colors = ['#A9BDD8', '#396386', '#A52B17', '#866446']
        titles = ['day1', 'day3']

        plt.figure(figsize=[27, 18])
        plt.suptitle(f'Mix days 2d by behavior trials- {self.mouse.name}', fontsize=30)

        for i_day in range(2):
            thirst, hunger = dot_products[i_day]
            relevant_thirst = [thirst[i] for i in range(len(thirst)) if colors[i_day][i] in good_colors]
            relevant_hunger = [hunger[i] for i in range(len(hunger)) if colors[i_day][i] in good_colors]
            relevant_colors = [c for c in colors[i_day] if c in good_colors]
            min_value = max(min(0, min([min(p) - 0.1 for p in [relevant_thirst, relevant_hunger]])), -2)
            max_value = min(max(1, max([max(p) + 0.1 for p in [relevant_thirst, relevant_hunger]])), 2)

            plt.subplot(1, 2, i_day + 1)
            plt.title(titles[i_day], fontsize=20)
            plt.scatter(
                relevant_thirst, relevant_hunger, c=relevant_colors, edgecolors='None', alpha=0.4, rasterized=True)
            plt.xlabel('Thirst', fontsize=20)
            plt.ylabel('Hunger', fontsize=20)
            plt.xlim(min_value, max_value)
            plt.ylim(min_value, max_value)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.gca().set_aspect('equal', adjustable='box')

        # plt.show()
        plt.savefig(join(PATH, f'{self.mouse.name}_mix_days_2d_by_behavior.jpg'))
        plt.close()

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
            colors = self.analyzer.classify_by_behavior_colors(
                day, products_dict[day.name], length_day, self.iti_size, before_behavior=True)
            if SMOOTH:
                axis_thirst = uniform_filter1d(axis_thirst, size=self.flat_parameter)
                axis_hunger = uniform_filter1d(axis_hunger, size=self.flat_parameter)
            dot_products.append([axis_thirst, axis_hunger])
            colors_classification.append(colors)

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
