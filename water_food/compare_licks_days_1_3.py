import numpy as np
import pandas as pd
from os.path import join, isdir
import matplotlib.pyplot as plt
from multiprocessing import Pool
plt.rcParams['svg.fonttype'] = 'none'

from configurations import *
from mouse import Mouse
from Visualizer import Visualizer
from main_analyzer import MainAnalyzer

MICE = ALL_MICE
WEEK_TYPE = f'week1'
DAYS = f'week1_days'
PATH = join(RESULTS_PATH, 'water_food', 'correlations', 'normalized_trials_1_3')


class RewardsCorrelationStability:

    def __init__(self, mouse):
        self.mouse = mouse
        self.days = self.mouse.days
        self.data_dir = mouse.week.data_dir_path
        self.sec_hz = mouse.smooth_factor
        self.dict_onsets = join(self.data_dir, 'rewards_indexes_onsets.npy')
        self.visualizer = Visualizer(self.sec_hz)
        self.main_analyzer = MainAnalyzer(self.sec_hz)

    def visualize(self, correlation_pairs):
        correlation_names = ['water original', 'water normalized', 'food original', 'food normalized']
        labels_names = [['water1', 'water3'], ['water1', 'water3'], ['food1', 'food3'], ['food1', 'food3']]
        plt.figure(figsize=(21, 21))
        plt.suptitle(f'{self.mouse.name} rewards - correlations licks normalized days 1-3', size=25)
        for i in range(len(correlation_names)):
            plt.subplot(2, 2, i + 1)
            self.visualizer.scatter_plot_config(correlation_pairs[i], labels_names[i], correlation_names[i])
        # plt.show()
        plt.savefig(join(PATH, f'{self.mouse.name}_correlation_reward_licks_comparison.jpg'))
        plt.close()

    def select_mean_responses(self, cell, index, onsets):
        all_trials = self.main_analyzer.load_trials_by_index(index, cell)
        bl_sub = all_trials.sub(all_trials.iloc[:, :self.sec_hz * 2].mean(axis=1), axis=0)
        rewards_responses = pd.DataFrame([bl_sub.iloc[t][onsets[t]: onsets[t] + 2 * self.sec_hz].values
                                          for t in range(len(bl_sub))])
        return rewards_responses

    def individual_licks(self, trial):
        new_trial = np.zeros(len(trial), dtype=int)
        licks = np.where(np.array(trial) == 1)[0]
        if len(licks) == 0:
            return 0
        first_licks = np.array([i for i in licks if i - 1 not in licks])
        new_trial[first_licks] = 1
        return sum(new_trial)

    def analyze_rewards(self, r_type, onsets_dict):
        responses_1, responses_3 = [], []
        data_1 = self.days[0].load_data_dict()
        cells_1 = self.main_analyzer.min_max_normalization(data_1['cells'])
        data_3 = self.days[2].load_data_dict()
        cells_3 = self.main_analyzer.min_max_normalization(data_3['cells'])

        if r_type == 'food':
            index_1 = onsets_dict['day1']['index_ate_full']
            latency_1 = onsets_dict['day1']['onsets_ate_full']
            index_3 = onsets_dict['day3']['index_ate_full']
            latency_3 = onsets_dict['day3']['onsets_ate_full']
            licks_1 = [self.individual_licks(data_1['pellet_licks'][v][latency_1[i] - self.sec_hz: latency_1[i]])
                       for i, v in enumerate(index_1)]
            licks_3 = [self.individual_licks(data_3['pellet_licks'][v][latency_3[i] - self.sec_hz: latency_3[i]])
                       for i, v in enumerate(index_3)]
        else:
            index_1 = onsets_dict['day1']['index_drank']
            latency_1 = onsets_dict['day1']['onsets_drank']
            index_3 = onsets_dict['day3']['index_drank']
            latency_3 = onsets_dict['day3']['onsets_drank']
            licks_1 = [self.individual_licks(data_1['water_licks'][v][latency_1[i] - self.sec_hz: latency_1[i] + self.sec_hz])
                       for i, v in enumerate(index_1)]
            licks_3 = [self.individual_licks(data_3['water_licks'][v][latency_3[i] - self.sec_hz: latency_3[i] + self.sec_hz])
                       for i, v in enumerate(index_3)]

        if len(index_1) < 5 or len(index_3) < 5:
            return [], []

        with Pool() as pool:
            rewards_1 = pool.starmap(self.select_mean_responses,
                                     [(cell, index_1, latency_1) for cell in cells_1])
            rewards_3 = pool.starmap(self.select_mean_responses,
                                     [(cell, index_3, latency_3) for cell in cells_3])

        relevant_trials_1 = []
        relevant_trials_3 = []
        if r_type == 'food':
            for i_trial in range(len(index_1)):
                timing = licks_1[i_trial]
                if timing == 0:
                    continue
                relevant_day3 = [i for i, v in enumerate(licks_3) if timing == v]
                if len(relevant_day3) > 0:
                    relevant_trials_3.append(relevant_day3)
                    relevant_trials_1.append(i_trial)

            if len(relevant_trials_3) > 5 and len(relevant_trials_1) > 5:
                for cell in range(len(cells_1)):
                    cell_day1_response = pd.DataFrame([rewards_1[cell].iloc[i] for i in relevant_trials_1]).mean(axis=0).mean()
                    final_cell_day3_response = []
                    for trials_group in relevant_trials_3:
                        final_cell_day3_response.append(pd.DataFrame([rewards_3[cell].iloc[i] for i in trials_group]).mean(axis=0))
                    # all_3 = list(set(np.hstack(relevant_trials_3)))
                    # cell_day3_response = pd.DataFrame([rewards_3[cell].iloc[i] for i in all_3]).mean(axis=0).mean()
                    cell_day3_response = pd.DataFrame(final_cell_day3_response).mean(axis=0).mean()
                    responses_1.append(cell_day1_response)
                    responses_3.append(cell_day3_response)
        else:
            for i_trial in range(len(index_3)):
                timing = licks_3[i_trial]
                relevant_day1 = [i for i, v in enumerate(licks_1) if timing == v]
                if len(relevant_day1) > 0:
                    relevant_trials_1.append(relevant_day1)
                    relevant_trials_3.append(i_trial)

            if len(relevant_trials_3) > 5 and len(relevant_trials_1) > 5:
                for cell in range(len(cells_1)):
                    cell_day3_response = pd.DataFrame([rewards_3[cell].iloc[i] for i in relevant_trials_3]).mean(
                        axis=0).mean()
                    final_cell_day1_response = []
                    for trials_group in relevant_trials_1:
                        final_cell_day1_response.append(
                            pd.DataFrame([rewards_1[cell].iloc[i] for i in trials_group]).mean(axis=0))
                    cell_day1_response = pd.DataFrame(final_cell_day1_response).mean(axis=0).mean()
                    # all_1 = list(set(np.hstack(relevant_trials_1)))
                    # cell_day1_response = pd.DataFrame([rewards_1[cell].iloc[i] for i in all_1]).mean(axis=0).mean()
                    responses_1.append(cell_day1_response)
                    responses_3.append(cell_day3_response)

        return responses_1, responses_3

    def run(self):
        full_responses = np.load(join(self.data_dir, 'dict_mean_responses.npy'), allow_pickle=True)[()]
        full_day1_food = full_responses['day_1']['food_reward']
        full_day3_food = full_responses['day_3']['food_reward']
        full_day1_water = full_responses['day_1']['water_reward']
        full_day3_water = full_responses['day_3']['water_reward']

        onsets = np.load(self.dict_onsets, allow_pickle=True)[()]
        water_1, water_3 = self.analyze_rewards('water', onsets)
        food_1, food_3 = self.analyze_rewards('food', onsets)

        correlations_pairs = [
            [full_day1_water, full_day3_water], [water_1, water_3], [full_day1_food, full_day3_food], [food_1, food_3]]

        filtered_pairs = []
        for pair in correlations_pairs:
            if len(pair[0]) == 0 or len(pair[1]) == 0:
                filtered_pairs.append([[], []])
                continue
            outlines = [i for i in range(len(pair[0])) if abs(pair[0][i]) > 1 or abs(pair[1][i]) > 1]
            filtered_day1 = [pair[0][i] for i in range(len(pair[0])) if i not in outlines]
            filtered_day2 = [pair[1][i] for i in range(len(pair[1])) if i not in outlines]
            filtered_pairs.append([filtered_day1, filtered_day2])

        self.visualize(filtered_pairs)

        all_correlations = [np.corrcoef(pair[0], pair[1])[0, 1] if len(pair[0]) and len(pair[1]) else np.nan
                            for pair in filtered_pairs]

        all_slopes = [np.polyfit(pair[0], pair[1], 1)[0] if len(pair[0]) and len(pair[1]) else np.nan
                      for pair in filtered_pairs]

        print(f'finished figures creation for {self.mouse.name}')
        return all_correlations, all_slopes


def barplot_config(responses, x_labels, colors, title):
    mice_by_condition = [[m[c] for m in responses] for c in range(len(responses[0]))]
    mice_mean = [np.nanmean(m) for m in mice_by_condition]
    plt.title(title, size=25)
    plt.bar(x_labels, mice_mean, color=colors)
    for m in responses:
        plt.plot(x_labels, m, c='gray', alpha=0.3)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=20)


def create_bar_plots(all_correlations, all_slopes):
    x_labels = ['water original', 'water normalized', 'food original', 'food normalized']
    colors = ['darkblue', 'blue', 'darkred', 'red']

    plt.figure(figsize=(30, 21))
    plt.suptitle('Summary correlation normalized days 1-3', size=30)

    plt.subplot(1, 2, 1)
    barplot_config(all_correlations, x_labels, colors, 'correlations')
    plt.subplot(1, 2, 2)
    barplot_config(all_slopes, x_labels, colors, 'slopes')

    plt.savefig(join(PATH, 'summary_normalized_behavior_days_1_3.jpg'))
    plt.savefig(join(PATH, 'summary_normalized_behavior_days_1_3.svg'))
    plt.close()
    # plt.show()


def main():
    all_correlations = []
    all_slopes = []
    for mouse_dict in ALL_MICE:
        if WEEK_TYPE in mouse_dict.keys():
            mouse = Mouse(mouse_dict, mouse_dict[DAYS], mouse_dict[WEEK_TYPE])
            processor = RewardsCorrelationStability(mouse)
            corr, slope = processor.run()
            processor.run()
            all_correlations.append(corr)
            all_slopes.append(slope)
    create_bar_plots(all_correlations, all_slopes)


'__main__' == __name__ and main()
