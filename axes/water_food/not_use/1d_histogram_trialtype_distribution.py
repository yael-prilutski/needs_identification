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
WEEK = 'week1'
DAYS = 'week1_days'
DICT_NAME = 'dot_products_dict.npy'
PATH = join(RESULTS_PATH, 'axes', 'water_food', 'mix_days_1d_behavior', 'behavior_before_activity', 'histogram_trial_type')


class MixedDays1dBehavior(Processor):

    def __init__(self, mouse, dict_name, *args, **kwargs):
        Processor.__init__(self, *args, **kwargs)
        self.mouse = mouse
        self.days = mouse.days
        self.week = mouse.week
        self.flat_parameter = int(3 * mouse.smooth_factor)
        self.analyzer = AxesAnalyzer(mouse.smooth_factor)
        self.iti_size = self.analyzer.iti_slice
        self.n_bins = 10
        self.dict_path = join(self.week.data_dir_path, 'axes', f'{self.mouse.name}_{dict_name}')
        self.trial_types_dict = {
            'darkblue': 'drank',
            'lightblue': 'not_drank',
            'darkred': 'ate',
            'lightcoral': 'not_ate',
            'green': 'omission_fluid',
            'gray': 'neutral',
            'purple': 'omission_pellet',
            'black': 'problem'
        }

    def config_hist_percentages(self, values, label, color, x_values, n_dots):
        if len(values) == 0:
            return None
        stable_bins = np.linspace(x_values[0], x_values[1], self.n_bins + 1, endpoint=True)
        counts, bins = np.histogram(values, bins=stable_bins)
        percentages = counts / n_dots * 100
        plt.plot(
            np.linspace(x_values[0], x_values[1], self.n_bins), percentages, alpha=0.5, label=label, color=color,
            linewidth=4)
        return percentages

    def visualize(self, dot_products):
        titles = ['day1 - Thirst', 'day1 - Hunger', 'day3 - Thirst', 'day3 - Hunger']

        plt.figure(figsize=[36, 21])
        plt.suptitle(f'Mix days trials type distribution- {self.mouse.name}', fontsize=30)

        for i in range(4):
            dict_trials = dot_products[i]
            n_dots = sum([len(v) for v in dict_trials.values()])
            min_value = min(min([min(v) for v in dict_trials.values() if len(v)]) - 0.1, 0)
            max_value = max(max([max(v) for v in dict_trials.values() if len(v)]) + 0.1, 1)

            plt.subplot(2, 2, i + 1)
            plt.title(titles[i], fontsize=20)
            for trial_type in dict_trials.keys():
                self.config_hist_percentages(dict_trials[trial_type], self.trial_types_dict[trial_type], trial_type,
                                             [min_value, max_value], n_dots)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.legend(fontsize=15)

        # plt.show()
        plt.savefig(join(PATH, f'{self.mouse.name}_mix_days_trials_type_hist.jpg'))
        plt.close()

    def rearrange_by_trial_type(self, colors, dot_product):
        distributions = {}
        for color in self.trial_types_dict.keys():
            trial_type_indices = np.where(colors == color)[0]
            distributions[color] = [dot_product[i] for i in trial_type_indices]
        return distributions

    def run(self):
        products_dict = np.load(self.dict_path, allow_pickle=True)[()]
        relevant_days = [self.days[0], self.days[2]]

        dot_products = []
        for i, day in enumerate(relevant_days):
            last_trial = products_dict[day.name]['last_vector_trial']
            length_day = np.where(products_dict[day.name]['relevant_trials'] == last_trial)[0][0] * self.iti_size
            axis_thirst = products_dict[day.name]['water_ortho'][:length_day]
            axis_hunger = products_dict[day.name]['food_ortho'][:length_day]
            day_colors = np.array(self.analyzer.classify_by_behavior_colors(
                day, products_dict[day.name], length_day, self.iti_size))
            dot_products.extend([
                self.rearrange_by_trial_type(day_colors, axis_thirst),
                self.rearrange_by_trial_type(day_colors, axis_hunger)])

        self.visualize(dot_products)
        return dot_products


def main():
    for mouse_p in RELEVANT_MICE:
        if WEEK in mouse_p.keys():
            mouse = Mouse(mouse_p, mouse_p[DAYS], mouse_p[WEEK])
            process = MixedDays1dBehavior(mouse, dict_name=DICT_NAME)
            process.run()
            print(f'finished processing {mouse.name}')


'__main__' == __name__ and main()
