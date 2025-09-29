import numpy as np
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from configurations import *
from mouse import Mouse
from main_analyzer import MainAnalyzer
from Visualizer import Visualizer

ANALYSIS_TYPE = 'reward'
MICE = ALL_MICE
WEEK_MIX = f'week1'
DAYS_MIX = f'week1_days'
WEEK_TYPE_WATER = 'opto_water_week'
WEEK_TYPE_FOOD = 'opto_food_week'
DAYS_WATER = 'opto_water_days'
DAYS_FOOD = 'opto_food_days'
PATH = join(RESULTS_PATH, 'water_food', 'modulation_index', 'between_weeks', ANALYSIS_TYPE)

THRESH_MI = 0.7
N_BINS = 10


class ModulationIndexBetween:

    def __init__(self, mouse, reward):
        self.mouse = mouse
        self.days = self.mouse.days
        self.week = self.mouse.week
        self.sec_hz = mouse.smooth_factor
        self.thresh_mi = THRESH_MI
        self.reward = reward
        self.main_analyzer = MainAnalyzer(self.sec_hz)
        self.visualizer = Visualizer(self.sec_hz)

    def visualize(self, values):
        dict_visualization = {
            'water': ['Water week', ['water_1', 'water_2'], 'darkblue'],
            'food': ['Food week', ['food_3', 'food_4'], 'darkred'],
            'water-food': ['Water-food week', ['water_2', 'food_4'], 'gold']
        }

        title, labels, color = dict_visualization[self.reward]
        std = np.std(values[1])

        plt.figure(figsize=(21, 21))
        plt.suptitle(f'{self.mouse.name} {title} std: {round(std, 2)}', size=30)

        plt.subplot(2, 1, 1)
        self.visualizer.scatter_plot_config(
            values[0], labels, 'Corr', additional_title=f'\nPerc similar: {round(values[2], 2)}')

        plt.subplot(2, 2, 3)
        plt.title('opposite values', size=25)
        plt.bar(['opposite_sign'], [values[3]], color=[color])

        plt.subplot(2, 2, 4)
        percentages = self.visualizer.config_hist_percentages(
            values[1], self.reward, color, None, len(values[0][0]), n_bins=N_BINS)
        # plt.show()
        plt.savefig(join(PATH, f'{self.reward}_{self.mouse.name}.jpg'))
        plt.close()
        return percentages, values[3], std

    def run(self):
        dict_responses = np.load(join(self.week.data_dir_path, 'dict_mean_responses.npy'), allow_pickle=True)[()]

        if self.reward == 'water-food':
            day1 = dict_responses['day_2'][f'water_{ANALYSIS_TYPE}']
            day2 = dict_responses['day_4'][f'food_{ANALYSIS_TYPE}']
        else:
            day1 = dict_responses['day_1'][ANALYSIS_TYPE]
            day2 = dict_responses['day_2'][ANALYSIS_TYPE]

        values = self.main_analyzer.modulation_index(day1, day2)
        all_hist, opposite, std_values = self.visualize(values)

        return all_hist, opposite, std_values


def anova_calculation(values):
    _, result_anova = f_oneway(values[0], values[1], values[2])
    values_combined = values[0] + values[1] + values[2]
    labels = ['water'] * len(values[0]) + ['food'] * len(values[1]) + ['both'] * len(values[2])

    tukey_std = pairwise_tukeyhsd(endog=values_combined, groups=labels, alpha=0.05)
    return [round(p, 3) for p in tukey_std.pvalues]


def summary_mice(water, food, mix):
    mean_hist = [np.array(pd.DataFrame([m[0] for m in response]).mean(axis=0)) for response in [water, food, mix]]
    std_hist = [np.array(pd.DataFrame([m[0] for m in response]).std(axis=0) / np.sqrt(len(response)))
                for response in [water, food, mix]]

    opposite_values = [[m[1] for m in mouse] for mouse in [water, food, mix]]
    barplot_opposite = [np.mean(d) for d in opposite_values]
    wf_f_opp, wf_w_opp, w_f_opp = anova_calculation(opposite_values)

    std_values = [[m[2] for m in trial_type] for trial_type in [water, food, mix]]
    barplot_std = [np.mean(d) for d in std_values]
    wf_f_std, wf_w_std, w_f_std = anova_calculation(std_values)

    x_labels = ['water', 'food', 'water-food']
    colors = ['darkblue', 'darkred', 'gold']

    plt.figure(figsize=(21, 21))
    plt.suptitle(f'Significant cells similarity and correlation', size=30)

    plt.subplot(2, 2, 1)
    plt.title(
        f'STD values: \nw-f: {round(w_f_std, 3)}, w-mix: {round(wf_w_std, 3)}, f-mix: {round(wf_f_std, 3)}', size=25)
    plt.bar(x_labels, barplot_std, color=['darkblue', 'darkred', 'gold'], alpha=0.7)
    plt.yticks(size=15)
    plt.xticks(size=15)
    for i in range(3):
        [plt.scatter(x_labels[i], val, color='black', alpha=0.5) for val in std_values[i]]

    plt.subplot(2, 2, 2)
    plt.title(f'Opposite sign: \nW-F: {round(w_f_opp, 3)}, W-mix: '
              f'{round(wf_w_opp, 3)}, F-mix: {round(wf_f_opp, 3)}', size=25)
    plt.bar(x_labels, barplot_opposite, color=['darkblue', 'darkred', 'gold'], alpha=0.7)
    plt.yticks(size=15)
    plt.xticks(size=15)
    for i in range(3):
        [plt.scatter(x_labels[i], val, color='black', alpha=0.5) for val in opposite_values[i]]

    plt.subplot(2, 1, 2)
    for i in range(3):
        plt.plot(np.arange(len(mean_hist[i])), mean_hist[i], color=colors[i], label=x_labels[i])
        plt.fill_between(
            np.arange(
                len(mean_hist[i])), mean_hist[i] - std_hist[i], mean_hist[i] + std_hist[i], color=colors[i], alpha=0.2)
    plt.xticks(np.linspace(0, N_BINS - 1, N_BINS - 1), np.linspace(-1, 1, N_BINS - 1), fontsize=15)
    plt.legend()

    plt.savefig(join(PATH, 'mice_summary.svg'))


def main():
    all_water = []
    all_food = []
    all_mix = []
    for mouse_dict in MICE:
        if WEEK_MIX in mouse_dict.keys():
            mouse = Mouse(mouse_dict, mouse_dict[DAYS_MIX], mouse_dict[WEEK_MIX])
            processor = ModulationIndexBetween(mouse, 'water-food')
            all_mix.append(processor.run())
        if WEEK_TYPE_WATER in mouse_dict.keys():
            mouse = Mouse(mouse_dict, mouse_dict[DAYS_WATER], mouse_dict[WEEK_TYPE_WATER])
            processor = ModulationIndexBetween(mouse, 'water')
            all_water.append(processor.run())
        if WEEK_TYPE_FOOD in mouse_dict.keys():
            mouse = Mouse(mouse_dict, mouse_dict[DAYS_FOOD], mouse_dict[WEEK_TYPE_FOOD])
            processor = ModulationIndexBetween(mouse, 'food')
            all_food.append(processor.run())

    summary_mice(all_water, all_food, all_mix)


'__main__' == __name__ and main()

