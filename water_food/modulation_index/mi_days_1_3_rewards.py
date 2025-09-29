import numpy as np
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from scipy.stats import ttest_ind

from configurations import *
from mouse import Mouse
from main_analyzer import MainAnalyzer
from Visualizer import Visualizer

ANALYSIS_TYPE = 'reward'
MICE = ALL_MICE
WEEK = f'week1'
DAYS = f'week1_days'
PATH = join(RESULTS_PATH, 'water_food', 'modulation_index', 'days_1_3', 'rewards_by_need', ANALYSIS_TYPE)

THRESH_MI = 0.7
N_BINS = 10


class ModulationIndexWithin:

    def __init__(self, mouse):
        self.mouse = mouse
        self.days = self.mouse.days
        self.week = self.mouse.week
        self.sec_hz = mouse.smooth_factor
        self.thresh_mi = THRESH_MI
        self.main_analyzer = MainAnalyzer(self.sec_hz)
        self.visualizer = Visualizer(self.sec_hz)

    def visualize(self, water, food):
        all_perc = []
        all_pairs = [water, food]
        colors = ['darkblue', 'darkred']
        labels_names = [['water_1', 'water_3'], ['food_1', 'food_3']]
        pairs_names = ['water_1_3', 'food_1_3']

        n_cells = len(food[0][0])
        mean_mi = [round(np.mean(p[1]), 2) if len(p[1]) > 0 else None for p in all_pairs]
        similar = [round(p[2], 2) if p[2] is not None else None for p in all_pairs]
        opposite = [p[3] if p[3] is not None else None for p in all_pairs]

        plt.figure(figsize=(21, 21))
        plt.suptitle(f'{self.mouse.name}\n day1: {mean_mi[0]}, day3: {mean_mi[1]}', size=30)

        y_hist = self.visualizer.highest_hist([water[1], food[1]], n_cells, n_bins=N_BINS)

        for i, plot in enumerate(all_pairs):
            plt.subplot(2, 2, i + 1)
            self.visualizer.scatter_plot_config(plot[0], labels_names[i], pairs_names[i],
                                                additional_title=f'\nPerc similar: {similar[i]}')

            plt.subplot(2, 2, i + 3)
            all_perc.append(self.visualizer.config_hist_percentages(
                plot[1], pairs_names[i], colors[i], y_hist, n_cells, n_bins=N_BINS))

        # plt.show()
        plt.savefig(join(PATH, f'{self.mouse.name}.jpg'))
        plt.close()
        return all_perc, opposite, mean_mi

    def run(self):
        dict_responses = np.load(join(self.week.data_dir_path, 'dict_mean_responses.npy'), allow_pickle=True)[()]

        day1_water = dict_responses['day_1'][f'water_{ANALYSIS_TYPE}']
        day1_food = dict_responses['day_1'][f'food_{ANALYSIS_TYPE}']
        day3_water = dict_responses['day_3'][f'water_{ANALYSIS_TYPE}']
        day3_food = dict_responses['day_3'][f'food_{ANALYSIS_TYPE}']

        water = self.main_analyzer.modulation_index(day1_water, day3_water)
        food = self.main_analyzer.modulation_index(day1_food, day3_food)

        all_hist, opposite, mean_values = self.visualize(water, food)

        print(f'finished figures creation for {self.mouse.name}')
        return all_hist, opposite, mean_values


def summary_mice(all_histogram, all_opposite, all_mean):
    len_relevant_hist = len([h for h in all_histogram if h[0] is not None])
    mean_hist = [np.array(pd.DataFrame([m[c] for m in all_histogram if m[c] is not None]).mean(axis=0)) for c in range(2)]
    std_hist = [np.array(pd.DataFrame([m[c] for m in all_histogram if m[c] is not None]
                                      ).std(axis=0) / np.sqrt(len_relevant_hist)) for c in range(2)]

    opposite_values = [[m[c] for m in all_opposite if m[c] is not None] for c in range(2)]
    barplot_opposite = [np.mean(r) for r in opposite_values]
    opposite_pvalue = ttest_ind(opposite_values[0], opposite_values[1])[1]

    mean_values = [[m[c] for m in all_mean if m[c] is not None] for c in range(2)]
    barplot_mean = [np.mean(r) for r in mean_values]
    mean_pvalue = ttest_ind(mean_values[0], mean_values[1])[1]

    colors = ['darkblue', 'darkred']
    x_labels = ['water', 'food']

    plt.figure(figsize=(21, 21))
    plt.suptitle(f'Significant cells similarity and correlation', size=30)

    plt.subplot(2, 2, 1)
    plt.title(f'Opposite sign: {round(opposite_pvalue, 3)}', fontsize=25)
    plt.bar(x_labels, barplot_opposite, color=['darkblue', 'darkred'], alpha=0.7)
    for mouse in all_opposite:
        plt.scatter(x_labels, mouse, color='black', alpha=0.5)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.subplot(2, 2, 2)
    plt.title(f'Mean: {round(mean_pvalue, 3)}', fontsize=25)
    plt.bar(x_labels, barplot_mean, color=['darkblue', 'darkred'], alpha=0.7)
    for mouse in all_mean:
        plt.scatter(x_labels, mouse, color='black', alpha=0.5)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.subplot(2, 1, 2)
    for i in range(2):
        plt.plot(np.arange(len(mean_hist[i])), mean_hist[i], color=colors[i], label=x_labels[i])
        plt.fill_between(np.arange(len(mean_hist[i])), mean_hist[i] - std_hist[i], mean_hist[i] + std_hist[i],
                         color=colors[i], alpha=0.2)
    plt.xticks(np.linspace(0, N_BINS - 1, N_BINS - 1), np.linspace(-1, 1, N_BINS - 1), fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend()

    plt.savefig(join(PATH, 'mice_summary.svg'))


def main():
    all_histogram = []
    all_opposite = []
    all_mean = []
    for mouse_dict in MICE:
        if WEEK in mouse_dict.keys():
            mouse = Mouse(mouse_dict, mouse_dict[DAYS], mouse_dict[WEEK])
            processor = ModulationIndexWithin(mouse)
            hist, opposite, mean = processor.run()
            all_histogram.append(hist)
            all_opposite.append(opposite)
            all_mean.append(mean)

    summary_mice(all_histogram, all_opposite, all_mean)


'__main__' == __name__ and main()

