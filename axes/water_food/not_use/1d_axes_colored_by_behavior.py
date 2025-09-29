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
SMOOTH = False
WEEK = 'week2'
DAYS = 'week2_days'
DICT_NAME = 'dot_products_dict.npy'
PATH = join(RESULTS_PATH, 'axes', 'mix_days_1d_behavior')
if BEFORE_BEHAVIOR:
    PATH = join(PATH, 'behavior_before_activity')
else:
    PATH = join(PATH, 'behavior_after_activity')
if not SMOOTH:
    PATH = join(PATH, 'no_smooth')

makedirs(PATH, exist_ok=True)


class MixedDays1dBehavior(Processor):

    def __init__(self, mouse, dict_name, *args, **kwargs):
        Processor.__init__(self, *args, **kwargs)
        self.mouse = mouse
        self.days = mouse.days
        self.week = mouse.week
        self.flat_parameter = int(3 * mouse.smooth_factor)
        self.analyzer = AxesAnalyzer(mouse.smooth_factor)
        self.iti_size = self.analyzer.iti_slice
        self.dict_path = join(self.week.data_dir_path, 'axes', f'{self.mouse.name}_{dict_name}')

    def visualize(self, dot_products, colors):
        titles = ['day1 - Thirst', 'day1 - Hunger', 'day3 - Thirst', 'day3 - Hunger']
        axes_names = ['Thirst', 'Hunger'] * 2

        plt.figure(figsize=[36, 21])
        plt.suptitle(f'Mix days 1d by behavior neutral trials- {self.mouse.name}', fontsize=30)

        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.title(titles[i], fontsize=20)
            plt.scatter(np.arange(len(dot_products[i])), dot_products[i], c=colors[i], edgecolors='None')
            plt.ylabel(axes_names[i], fontsize=20)
            plt.ylim(min(0, min(dot_products[i]) - 0.1), max(1, max(dot_products[i]) + 0.1))
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)

        # plt.show()
        plt.savefig(join(PATH, f'{self.mouse.name}_mix_days_1d_be_behavior_neutral.jpg'))
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
            day_colors = self.analyzer.classify_by_behavior_colors(
                day, products_dict[day.name], length_day, self.iti_size, before_behavior=BEFORE_BEHAVIOR)
            if SMOOTH:
                axis_thirst = uniform_filter1d(axis_thirst, self.flat_parameter)
                axis_hunger = uniform_filter1d(axis_hunger, self.flat_parameter)
            dot_products.extend([axis_thirst, axis_hunger])
            colors_classification.extend([day_colors, day_colors])

        self.visualize(dot_products, colors_classification)
        return dot_products


def main():
    for mouse_p in RELEVANT_MICE:
        if WEEK in mouse_p.keys():
            mouse = Mouse(mouse_p, mouse_p[DAYS], mouse_p[WEEK])
            process = MixedDays1dBehavior(mouse, dict_name=DICT_NAME)
            process.run()
            print(f'finished processing {mouse.name}')


'__main__' == __name__ and main()
