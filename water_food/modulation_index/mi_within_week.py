import numpy as np
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
plt.rcParams['svg.fonttype'] = 'none'

from configurations import *
from mouse import Mouse
from main_analyzer import MainAnalyzer
from Visualizer import Visualizer

ANALYSIS_TYPE = 'reward'
MICE = ALL_MICE
WEEK = f'week1'
DAYS = f'week1_days'
PATH = join(RESULTS_PATH, 'water_food', 'modulation_index', 'within_week', ANALYSIS_TYPE)


N_BINS = 10


class ModulationIndexWithin:

    def __init__(self, mouse):
        self.mouse = mouse
        self.days = self.mouse.days
        self.week = self.mouse.week
        self.sec_hz = mouse.smooth_factor
        self.main_analyzer = MainAnalyzer(self.sec_hz)
        self.visualizer = Visualizer(self.sec_hz)

    def visualize(self, water, food, water_food):
        all_perc = []
        all_pairs = [water, food, water_food]
        colors = ['darkblue', 'darkred', 'gold']
        labels_names = [['water_2', 'water_1'], ['food_3', 'food_4'], ['water_2', 'food_4']]
        pairs_names = ['water_1_2', 'food_3_4', 'water2_food_4']

        n_cells = len(water[0][0])
        std_mi = [round(np.std(p[1]), 2) for p in all_pairs]
        similar = [round(p[2], 2) for p in all_pairs]
        opposite = [p[3] for p in all_pairs]

        plt.figure(figsize=(21, 21))
        plt.suptitle(f'{self.mouse.name}\n water: {std_mi[0]}, food: {std_mi[1]}, both: {std_mi[2]}', size=30)

        y_hist = self.visualizer.highest_hist([water[1], food[1], water_food[1]], n_cells, n_bins=N_BINS)

        for i, plot in enumerate(all_pairs):
            plt.subplot(2, 3, i + 1)
            self.visualizer.scatter_plot_config(plot[0], labels_names[i], pairs_names[i],
                                                additional_title=f'\nPerc similar: {similar[i]}')

            plt.subplot(2, 3, i + 4)
            all_perc.append(self.visualizer.config_hist_percentages(
                plot[1], pairs_names[i], colors[i], y_hist, n_cells, n_bins=N_BINS))

        # plt.show()
        plt.savefig(join(PATH, f'{self.mouse.name}.jpg'))
        plt.close()
        return all_perc, opposite, std_mi

    def run(self):
        dict_responses = np.load(join(self.week.data_dir_path, 'dict_mean_responses.npy'), allow_pickle=True)[()]

        day1 = dict_responses['day_1'][f'water_{ANALYSIS_TYPE}']
        day2 = dict_responses['day_2'][f'water_{ANALYSIS_TYPE}']
        day3 = dict_responses['day_3'][f'food_{ANALYSIS_TYPE}']
        day4 = dict_responses['day_4'][f'food_{ANALYSIS_TYPE}']

        water_values = self.main_analyzer.modulation_index(day1, day2)
        food_values = self.main_analyzer.modulation_index(day3, day4)
        water_food_values = self.main_analyzer.modulation_index(day2, day4)

        all_hist, opposite, std_values = self.visualize(water_values, food_values, water_food_values)

        print(f'finished figures creation for {self.mouse.name}')
        return all_hist, opposite, std_values


def calculate_anova(mice_values):
    labels = ['water', 'food', 'both'] * len(mice_values)
    data_std = pd.DataFrame({
        'mouse': np.hstack([[str(i)] * 3 for i in range(len(mice_values))]),
        'time': labels,
        'values': np.hstack(mice_values)
    })

    rm_anova = AnovaRM(data_std, depvar='values', subject='mouse', within=['time']).fit()
    tukey_std = pairwise_tukeyhsd(endog=np.hstack(mice_values), groups=labels, alpha=0.05)
    return [round(p, 4) for p in tukey_std.pvalues]


def summary_mice(all_histogram, all_opposite, all_std):
    mean_hist = [np.array(pd.DataFrame([m[c] for m in all_histogram]).mean(axis=0)) for c in range(3)]
    std_hist = [np.array(pd.DataFrame([m[c] for m in all_histogram]).std(axis=0) / np.sqrt(len(all_histogram)))
                for c in range(3)]

    opposite_values = [[m[c] for m in all_opposite] for c in range(3)]
    barplot_opposite = [np.mean(r) for r in opposite_values]
    wf_f_opp, wf_w_opp, w_f_opp = calculate_anova(all_opposite)

    std_values = [[m[c] for m in all_std] for c in range(3)]
    barplot_std = [np.mean(r) for r in std_values]
    wf_f_std, wf_w_std, w_f_std = calculate_anova(all_std)

    x_labels = ['water', 'food', 'water-food']
    colors = ['darkblue', 'darkred', 'gold']

    plt.figure(figsize=(21, 21))
    plt.suptitle(f'Significant cells similarity and correlation', size=30)

    plt.subplot(2, 2, 1)
    plt.title(f'Water-food STD: \nW-F: {w_f_std}, W-mix: {wf_w_std}, F-mix: {wf_f_std}',
              fontsize=25)
    plt.bar(x_labels, barplot_std, color=['darkblue', 'darkred', 'gold'], alpha=0.7)
    plt.yticks(size=15)
    for mouse in all_std:
        plt.scatter(x_labels, mouse, color='black', alpha=0.5)

    plt.subplot(2, 2, 2)
    plt.title(f'Opposite sign: \nW-F: {round(w_f_opp, 2)}, W-mix: '
              f'{round(wf_w_opp, 2)}, F-mix: {round(wf_f_opp, 2)}', fontsize=25)
    plt.bar(x_labels, barplot_opposite, color=['darkblue', 'darkred', 'gold'], alpha=0.7)
    for mouse in all_opposite:
        plt.scatter(x_labels, mouse, color='black', alpha=0.5)

    plt.subplot(2, 1, 2)

    for i in range(3):
        plt.plot(np.arange(len(mean_hist[i])), mean_hist[i], color=colors[i], label=x_labels[i])
        plt.fill_between(np.arange(len(mean_hist[i])), mean_hist[i] - std_hist[i], mean_hist[i] + std_hist[i],
                         color=colors[i], alpha=0.2)
    plt.xticks(np.linspace(0, N_BINS - 1, N_BINS - 1), np.linspace(-1, 1, N_BINS - 1), fontsize=15)
    plt.legend()

    # plt.show()
    plt.savefig(join(PATH, 'mice_summary.jpg'))


def main():
    all_histogram = []
    all_opposite = []
    all_std = []
    for mouse_dict in MICE:
        if WEEK in mouse_dict.keys():
            mouse = Mouse(mouse_dict, mouse_dict[DAYS], mouse_dict[WEEK])
            processor = ModulationIndexWithin(mouse)
            hist, opposite, std = processor.run()
            all_histogram.append(hist)
            all_opposite.append(opposite)
            all_std.append(std)

    summary_mice(all_histogram, all_opposite, all_std)


'__main__' == __name__ and main()

