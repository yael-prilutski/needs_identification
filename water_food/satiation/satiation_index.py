import numpy as np
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
plt.rcParams['svg.fonttype'] = 'none'

from configurations import *
from mouse import Mouse
from Visualizer import Visualizer

ANALYSIS_TYPE = 'reward'
PATH = join(RESULTS_PATH, 'water_food', 'satiation', 'satiation_index', ANALYSIS_TYPE)


N_BINS = 10
N_TRIALS = 30


class ModulationIndexWithin:

    def __init__(self, mouse, reward):
        self.mouse = mouse
        self.days = self.mouse.days
        self.week = self.mouse.week
        self.reward = reward
        if reward == 'water':
            self.color = 'darkblue'
        else:
            self.color = 'darkred'
        self.sec_hz = mouse.smooth_factor
        self.n_trials = N_TRIALS
        self.visualizer = Visualizer(self.sec_hz)

    def visualize(self, responses, start_day_responses):
        n_cells = len(responses[0][0])
        std_si = round(np.std(responses[1]), 2)
        std_si2 = round(np.std(start_day_responses[1]), 2)
        opposite = responses[2]
        opposite2 = start_day_responses[2]

        plt.figure(figsize=(21, 21))
        plt.suptitle(f'{self.mouse.name}\n std: {std_si}', size=30)

        plt.subplot(2, 2, 1)
        slope = round(np.polyfit(responses[0][0], responses[0][1], 1)[0], 2)
        corr = np.corrcoef(responses[0][0], responses[0][1])[0, 1]
        self.visualizer.scatter_plot_config(responses[0], ['start', 'end'], self.reward,
                                            additional_title=f', slope: {slope }\nPerc opposite: {opposite}')

        plt.subplot(2, 2, 2)
        slope2 = round(np.polyfit(start_day_responses[0][0], start_day_responses[0][1], 1)[0], 2)
        corr2 = np.corrcoef(start_day_responses[0][0], start_day_responses[0][1])[0, 1]
        self.visualizer.scatter_plot_config(
            start_day_responses[0], ['day1 - start', 'day2 - start'], f'{self.reward} 2 days',
            additional_title=f', slope: {slope2}\nPerc opposite: {opposite2}')

        plt.subplot(2, 2, 3)
        perc = self.visualizer.config_hist_percentages(
            responses[1], self.reward, self.color, None, n_cells, n_bins=N_BINS)

        plt.subplot(2, 2, 4)
        perc2 = self.visualizer.config_hist_percentages(
            start_day_responses[1], f'{self.reward} 2 days', self.color, None,  len(start_day_responses[0][0]),
            n_bins=N_BINS)

        # plt.show()
        plt.savefig(join(PATH, f'{self.reward}_{self.mouse.name}.jpg'))
        plt.close()
        return perc, [opposite, opposite2], [std_si, std_si2], [corr, corr2], [slope, slope2]

    def satiation_index(self, responses, responses2):
        start_v, end_v, si, opposite_cells, perc_opposite = [], [], [], [], []
        for cell in range(len(responses)):
            if len(responses2) == 0:
                start = np.mean(responses[cell][:self.n_trials])
                end = np.mean(responses[cell][-self.n_trials:])
            else:
                start = np.mean(responses[cell][:self.n_trials])
                end = np.mean(responses2[cell][:self.n_trials])
            start_v.append(start)
            end_v.append(end)

            if start * end < 0:
                if abs(start / end) >= 1.5:
                    end = 0
                elif abs(end / start) >= 1.5:
                    start = 0
                if start != 0 and end != 0:
                    opposite_cells.append(cell)
                    continue
            si.append((start - end) / (start + end))

        perc_opposite = round(len(opposite_cells) / len(responses) * 100, 2)
        return [start_v, end_v], si, perc_opposite

    def run(self):
        dict_responses = np.load(
            join(self.week.data_dir_path, 'dict_mean_responses_per_trial.npy'), allow_pickle=True)[()]
        responses = dict_responses[f'day1_{ANALYSIS_TYPE}']
        responses2 = dict_responses[f'day2_{ANALYSIS_TYPE}']

        si_values = self.satiation_index(responses, [])
        mi_2days_values = self.satiation_index(responses, responses2)

        all_hist, opposite, std_values, corr, slope = self.visualize(si_values, mi_2days_values)

        print(f'finished figures creation for {self.mouse.name}')
        return all_hist, opposite, std_values, corr, slope, [np.mean(si_values[1]), np.mean(mi_2days_values[1])]


def config_barplot(water, food, analysis_name, x_labels, colors):
    water_satiation, water_start = [[m[i] for m in water] for i in range(2)]
    food_satiation, food_start = [[m[i] for m in food] for i in range(2)]

    barplot_values = [np.mean(p) for p in [water_satiation, water_start, food_satiation, food_start]]
    ttest_water = round(ttest_rel(water_satiation, water_start)[1], 4)
    ttest_food = round(ttest_rel(food_satiation, food_start)[1], 4)
    plt.title(f'ttest {analysis_name}: \nwater: {ttest_water}, \nfood: {ttest_food}', fontsize=25)
    plt.bar(x_labels, barplot_values, color=colors, alpha=0.7)
    for mouse in water:
        plt.plot(x_labels[:2], [mouse[0], mouse[1]], color='gray', alpha=0.5)
    for mouse in food:
        plt.plot(x_labels[2:], [mouse[0], mouse[1]], color='gray', alpha=0.5)
    plt.yticks(size=15)
    plt.xticks(size=15)


def summary_mice(water, food):
    # all_histogram, all_opposite, all_std, all_corr, all_slope, all_mean
    mean_hist = [np.array(pd.DataFrame(reward[0]).mean(axis=0)) for reward in [water, food]]
    std_hist = [np.array(pd.DataFrame(reward[0]).std(axis=0) / np.sqrt(len(reward[0]))) for reward in [water, food]]

    x_labels = ['water\nsatiation', 'water\nstart', 'food\nsatiation', 'food\nstart']
    colors = ['darkblue', 'lightblue', 'darkred', 'coral']
    colors_hist = ['darkblue', 'darkred']

    plt.figure(figsize=(35, 18))
    plt.suptitle(f'Satiation index', size=30)

    plt.subplot(2, 5, 1)
    config_barplot(water[1], food[1], 'opposite', x_labels, colors)

    plt.subplot(2, 5, 2)
    config_barplot(water[2], food[2], 'std', x_labels, colors)

    plt.subplot(2, 5, 3)
    config_barplot(water[3], food[3], 'correlation', x_labels, colors)

    plt.subplot(2, 5, 4)
    config_barplot(water[4], food[4], 'slope', x_labels, colors)

    plt.subplot(2, 5, 5)
    config_barplot(water[5], food[5], 'mean', x_labels, colors)

    plt.subplot(2, 1, 2)
    for i in range(2):
        plt.plot(np.arange(len(mean_hist[i])), mean_hist[i], color=colors_hist[i], label=x_labels[i])
        plt.fill_between(np.arange(len(mean_hist[i])), mean_hist[i] - std_hist[i], mean_hist[i] + std_hist[i],
                         color=colors_hist[i], alpha=0.2)
    plt.xticks(np.linspace(0, N_BINS - 1, N_BINS - 1), np.linspace(-1, 1, N_BINS - 1), fontsize=15)
    plt.legend()

    # plt.show()
    plt.savefig(join(PATH, 'mice_summary.jpg'))


def main():
    water_responses, food_responses = [], []
    for reward, responses_list in [['water', water_responses], ['food', food_responses]]:
        days = f'opto_{reward}_days'
        week = f'opto_{reward}_week'
        all_histogram, all_opposite, all_std, all_corr, all_slope, all_mean = [], [], [], [], [], []
        for mouse_dict in ALL_MICE:
            if week in mouse_dict.keys():
                mouse = Mouse(mouse_dict, mouse_dict[days], mouse_dict[week])
                processor = ModulationIndexWithin(mouse, reward)
                hist, opposite, std, corr, slope, mean = processor.run()
                all_histogram.append(hist)
                all_opposite.append(opposite)
                all_std.append(std)
                all_corr.append(corr)
                all_slope.append(slope)
                all_mean.append(mean)
        responses_list.extend([all_histogram, all_opposite, all_std, all_corr, all_slope, all_mean])

    summary_mice(water_responses, food_responses)


'__main__' == __name__ and main()

