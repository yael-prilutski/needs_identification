import numpy as np
import pandas as pd
import seaborn as sns
from os.path import join
import matplotlib.pyplot as plt
from multiprocessing import Pool
from matplotlib.colors import TwoSlopeNorm
plt.rcParams['svg.fonttype'] = 'none'

from configurations import *
from mouse import Mouse
from Visualizer import Visualizer
from main_analyzer import MainAnalyzer

MICE = [MOUSE_YP83]
WEEK_TYPE = f'week1'
DAYS = f'week1_days'
BY_REWARD = True
PATH = join(RESULTS_PATH, 'water_food', 'rewards_heatmap')
if BY_REWARD:
    PATH = join(PATH, 'by_reward')
    SEC_PRE_TRIAL = 3
else:
    PATH = join(PATH, 'by_cue')
    SEC_PRE_TRIAL = 1


class HeatmapsRewards:

    def __init__(self, mouse):
        self.mouse = mouse
        self.dict_responses = join(mouse.week.data_dir_path, 'rewards_indexes_onsets.npy')
        self.days = self.mouse.days
        self.sec_hz = mouse.smooth_factor
        self.main_analyzer = MainAnalyzer(self.sec_hz)
        self.visualizer = Visualizer(self.sec_hz, sec_pre_trial=SEC_PRE_TRIAL)
        self.min_trials = 30

    def single_plot_config(self, df, color, title):
        plt.title(title, fontsize=25)
        if len(df):
            min_value = df.min().min() * 0.4
            max_value = df.max().max() * 0.4
            # sns.heatmap(df, cmap=color, vmin=min_value, vmax=max_value,
            #             norm=TwoSlopeNorm(vmin=min_value, vcenter=0, vmax=max_value))
            img = plt.imshow(df, cmap=color, norm=TwoSlopeNorm(vmin=min_value, vcenter=0, vmax=max_value),
                             aspect='auto', interpolation='nearest', rasterized=True)
            plt.colorbar(img)
            plt.axvline(x=3 * self.sec_hz, linewidth=1, color='black', linestyle='--')
            if not BY_REWARD:
                plt.axvline(x=1 * self.sec_hz, linewidth=1, color='black', linestyle='--')
            plt.xticks(fontsize=20)
            self.visualizer.set_xticks(len(df.iloc[0]))

    def visualize(self, all_days_responses, reward_type):
        color = sns.diverging_palette(255, 10, sep=120, n=100, as_cmap=True)

        plt.figure(figsize=(24, 24))
        plt.suptitle(f'{self.mouse.name} - by {reward_type}', size=30)
        for i, name in enumerate(['Water1', 'Water2', 'Water3', 'Food1', 'Food3', 'Food4']):
            plt.subplot(2, 3, i + 1)
            self.single_plot_config(all_days_responses[i], color, name)

        # plt.show()
        if BY_REWARD:
            name = f'{self.mouse.name}_{reward_type}_rewards_heatmaps.svg'
        else:
            name = f'{self.mouse.name}_{reward_type}_cues_heatmaps.jpg'
        plt.savefig(join(PATH, name))
        plt.close()

    def sort_responses(self, df):
        mean_values = df.iloc[:, round(self.sec_hz * 3): round(5 * self.sec_hz)].mean(axis=1)
        sorted_df = df.assign(mean=mean_values).sort_values('mean')
        sorted_df = sorted_df.drop('mean', axis=1)
        return sorted_df.index

    def allign_by_rewards(self, cell, index, onsets):
        all_trials = self.main_analyzer.load_trials_by_index(index, cell)
        bl_sub = all_trials.sub(all_trials.iloc[:, :self.sec_hz * 2].mean(axis=1), axis=0)
        rewards_responses = pd.DataFrame(
            [bl_sub.iloc[t][onsets[t] - 3 * self.sec_hz: onsets[t] + 4 * self.sec_hz].values
             for t in range(len(bl_sub))])
        return rewards_responses.mean(axis=0)

    def load_day_rewards(self, day_i, index_dict):
        day = self.days[day_i]
        data = day.load_data_dict()
        cells = self.main_analyzer.min_max_normalization(data['cells'])

        water_responses = []
        food_responses = []

        with Pool() as pool:
            if len(index_dict['index_ate_full']) > 5:
                if BY_REWARD:
                    onsets_food = index_dict['onsets_ate_full']
                else:
                    onsets_food = [self.sec_hz * 4] * len(index_dict['index_ate_full'])
                food_responses = pool.starmap(
                    self.allign_by_rewards, [(cell, index_dict['index_ate_full'][:self.min_trials],
                                              onsets_food[:self.min_trials]) for cell in cells])

            if len(index_dict['index_drank']) > 5:
                if BY_REWARD:
                    onsets_drank = index_dict['onsets_drank']
                else:
                    onsets_drank = [self.sec_hz * 4] * len(index_dict['index_drank'])
                water_responses = pool.starmap(
                    self.allign_by_rewards, [(cell, index_dict['index_drank'][:self.min_trials],
                                              onsets_drank[:self.min_trials]) for cell in cells])

        return pd.DataFrame(water_responses), pd.DataFrame(food_responses)

    def run(self):
        dict_indexes = np.load(self.dict_responses, allow_pickle=True)[()]
        day1, day2, day3, day4 = [self.load_day_rewards(day, dict_indexes[f'day{day + 1}']) for day in range(4)]
        relevant_responses = [day1[0], day2[0], day3[0], day1[1], day3[1], day4[1]]
        water_order = self.sort_responses(day2[0])
        reordered_by_water = [pd.DataFrame([response.iloc[i] for i in water_order])
                              if len(response) else [] for response in relevant_responses]
        food_order = self.sort_responses(day4[1])
        reordered_by_food = [pd.DataFrame([response.iloc[i] for i in food_order])
                             if len(response) else [] for response in relevant_responses]

        self.visualize(reordered_by_water, 'water')
        self.visualize(reordered_by_food, 'food')

        print(f'finished figures creation for {self.mouse.name}')


def main():
    for mouse_dict in MICE:
        if WEEK_TYPE in mouse_dict.keys():
            mouse = Mouse(mouse_dict, mouse_dict[DAYS], mouse_dict[WEEK_TYPE])
            processor = HeatmapsRewards(mouse)
            processor.run()


'__main__' == __name__ and main()
