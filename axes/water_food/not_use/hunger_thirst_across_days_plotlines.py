import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from os.path import join, isdir, dirname
from scipy.ndimage import uniform_filter1d

from configurations import *
from mouse import Mouse
from Processor import Processor
from axes.axes_analyzer import AxesAnalyzer


RELEVANT_MICE = ALL_MICE
DICT_NAME = 'dot_products_dict.npy'
PATH = join(RESULTS_PATH, 'axes', 'water_food', 'hunger_thirst_across_days_plotlines')
NO_ORTHOGONALIZATION = True
if NO_ORTHOGONALIZATION:
    PATH = join(PATH, 'no_ortho')


class BetweenDaysAxis(Processor):

    def __init__(self, mouse, dict_name, main_week, is_water=True, no_ortho=False, *args, **kwargs):
        Processor.__init__(self, *args, **kwargs)
        self.mouse = mouse
        self.days = mouse.days
        self.week = mouse.week
        self.main_week = main_week
        self.is_water = is_water
        self.no_ortho = no_ortho
        self.flat_parameter = int(20 * mouse.smooth_factor)
        self.iti_size = AxesAnalyzer(mouse.smooth_factor).iti_slice
        self.dict_path = join(self.week.data_dir_path, 'axes', f'{self.mouse.name}_{dict_name}')

    def visualize(self, dot_products):
        if self.main_week:
            suptitle = 'Ongoing activity water2-food4'
            titles = ['day2 - water', 'day4 - food']
            colors = ['darkblue', 'darkred']
            labels = ['axis-water', 'axis-food']
            name = f'water_food_{self.mouse.name}.jpg'
        elif self.is_water:
            suptitle = 'Ongoing activity water'
            titles = ['water1', 'water2']
            colors = ['darkblue', 'lightblue']
            labels = ['axis1', 'axis2']
            name = f'water_days_{self.mouse.name}.jpg'
        else:
            suptitle = 'Ongoing activity food'
            titles = ['food1', 'food2']
            colors = ['darkred', 'lightcoral']
            labels = ['axis1', 'axis2']
            name = f'food_days_{self.mouse.name}.jpg'

        plt.figure(figsize=[18, 12])
        plt.suptitle(f'{suptitle} {self.mouse.name}', fontsize=30)

        for i in range(2):
            plt.subplot(1, 2, i + 1)
            plt.title(titles[i], fontsize=20)
            day1, day2 = dot_products[i]
            plt.plot(day1, color=colors[0], label=labels[0])
            plt.plot(day2, color=colors[1], label=labels[1])
            min_y = min(min(day1) - 0.1, min(day2) - 0.1, -0.1)
            max_y = max(max(day1) + 0.1, max(day2) + 0.1, 1.1)
            plt.ylim(min_y, max_y)
            plt.legend()

        # plt.show()
        plt.savefig(join(PATH, name))
        plt.close()

    def run(self):
        products_dict = np.load(self.dict_path, allow_pickle=True)[()]
        if self.main_week:
            relevant_days = [self.days[1], self.days[3]]
        else:
            relevant_days = [day for day in self.days if day.name[-1] != 'b'][:2]

        dot_products = []
        for i, day in enumerate(relevant_days):
            last_trial = products_dict[day.name]['last_vector_trial']
            length_day = np.where(products_dict[day.name]['relevant_trials'] == last_trial)[0][0] * self.iti_size
            if self.main_week:
                if self.no_ortho:
                    axis1 = products_dict[day.name]['water'][:length_day]
                    axis2 = products_dict[day.name]['food'][:length_day]
                else:
                    axis1 = products_dict[day.name]['water_ortho'][:length_day]
                    axis2 = products_dict[day.name]['food_ortho'][:length_day]
            else:
                axis1 = products_dict[day.name]['vector_1'][:length_day]
                axis2 = products_dict[day.name]['vector_2'][:length_day]
            dot_products.append(
                [uniform_filter1d(axis1, self.flat_parameter), uniform_filter1d(axis2, self.flat_parameter)])

        self.visualize(dot_products)
        return dot_products


def resize_array(array, length):
    array_old = np.linspace(0, 1, len(array))
    array_new = np.linspace(0, 1, length)
    return np.interp(array_new, array_old, array)


def process_analyze_type(all_mice):
    day1_mice = [mouse[0] for mouse in all_mice]
    day2_mice = [mouse[1] for mouse in all_mice]

    length_day1 = int(np.percentile([len(mouse[0]) for mouse in day1_mice], 50))
    length_day2 = int(np.percentile([len(mouse[0]) for mouse in day2_mice], 50))

    wrapped_day1 = []
    for mouse in day1_mice:
        axis1, axis2 = mouse
        axis1_new = resize_array(axis1, length_day1)
        axis2_new = resize_array(axis2, length_day1)
        wrapped_day1.append([axis1_new, axis2_new])

    mean_day1 = [pd.DataFrame([mouse[i] for mouse in wrapped_day1]).mean(axis=0) for i in range(2)]
    std_day1 = [pd.DataFrame(
        [mouse[i] for mouse in wrapped_day1]).std(axis=0) / np.sqrt(len(all_mice)) for i in range(2)]

    wrapped_day2 = []
    for mouse in day2_mice:
        axis1, axis2 = mouse
        axis1_new = resize_array(axis1, length_day2)
        axis2_new = resize_array(axis2, length_day2)
        wrapped_day2.append([axis1_new, axis2_new])

    mean_day2 = [pd.DataFrame([mouse[i] for mouse in wrapped_day2]).mean(axis=0) for i in range(2)]
    std_day2 = [pd.DataFrame(
        [mouse[i] for mouse in wrapped_day2]).std(axis=0) / np.sqrt(len(all_mice)) for i in range(2)]

    return [mean_day1, std_day1], [mean_day2, std_day2]


def summary_mice(day2_4_axes, water_axes, food_axes):
    day2, day4 = process_analyze_type(day2_4_axes)
    water1, water2 = process_analyze_type(water_axes)
    food1, food2 = process_analyze_type(food_axes)

    all_activity = [day2, day4, water1, water2, food1, food2]

    titles = ['day2', 'day4', 'water1', 'water2', 'food1', 'food2']
    colors = [['darkblue', 'darkred'], ['darkblue', 'darkred'], ['blue', 'lightblue'], ['blue', 'lightblue'],
              ['red', 'lightcoral'], ['red', 'lightcoral']]
    labels = [['water axis', 'food axis'], ['water axis', 'food axis'], ['water1 axis', 'water2 axis'],
              ['water1 axis', 'water2 axis'], ['food1 axis', 'food2 axis'], ['food1 axis', 'food2 axis']]

    plt.figure(figsize=[24, 24])
    plt.suptitle(f'Ongoing activity water-food summary', fontsize=30)
    for i in range(6):
        mean, std = all_activity[i]
        plt.subplot(3, 2, i + 1)
        plt.title(titles[i], fontsize=20)
        plt.plot(mean[0], color=colors[i][0], label=labels[i][0])
        plt.fill_between(
            np.arange(len(mean[0])), mean[0] - std[0], mean[0] + std[0], alpha=0.5, color=colors[i][0])
        plt.plot(mean[1], color=colors[i][1], label=labels[i][1])
        plt.fill_between(
            np.arange(len(mean[1])), mean[1] - std[1], mean[1] + std[1], alpha=0.5, color=colors[i][1])
        plt.legend()

        plt.ylim(min(-0.1, min(mean[0]) - 0.1, min(mean[1]) - 0.1),
                 max(1.1, max(mean[0]) + 0.1, max(mean[1]) + 0.1))

    plt.savefig(join(PATH, f'summary_all_mice_water_food.jpg'))


def main():
    day2_4_axes = []
    water_axes = []
    food_axes = []
    for mouse_p in RELEVANT_MICE:
        if 'week1' in mouse_p.keys():
            mouse = Mouse(mouse_p, mouse_p['week1_days'], mouse_p['week1'])
            process = BetweenDaysAxis(mouse, dict_name=DICT_NAME, main_week=True, no_ortho=NO_ORTHOGONALIZATION)
            day2_4_axes.append(process.run())
            print(f'finished processing {mouse.name}')
    for mouse_p in RELEVANT_MICE:
        if 'opto_water_week' in mouse_p.keys():
            mouse = Mouse(mouse_p, mouse_p['opto_water_days'], mouse_p['opto_water_week'])
            process = BetweenDaysAxis(mouse, dict_name=DICT_NAME, main_week=False)
            water_axes.append(process.run())
            print(f'finished processing {mouse.name}')
    for mouse_p in RELEVANT_MICE:
        if 'opto_food_week' in mouse_p.keys():
            mouse = Mouse(mouse_p, mouse_p['opto_food_days'], mouse_p['opto_food_week'])
            process = BetweenDaysAxis(mouse, dict_name=DICT_NAME, main_week=False, is_water=False)
            food_axes.append(process.run())
            print(f'finished processing {mouse.name}')
    summary_mice(day2_4_axes, water_axes, food_axes)


'__main__' == __name__ and main()
