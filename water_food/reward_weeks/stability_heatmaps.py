import numpy as np
import pandas as pd
from os.path import join, isfile
from multiprocessing import Pool
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
plt.rcParams['svg.fonttype'] = 'none'

from configurations import *
from mouse import Mouse
from main_analyzer import MainAnalyzer

PATH = join(RESULTS_PATH, 'water_food', 'stability', 'heatmaps')


class StabilityHeatmaps:

    def __init__(self, mouse, reward):
        self.mouse = mouse
        self.days = [d for d in mouse.days if d.name[-1] != 'b']
        self.data_dir = mouse.week.data_dir_path
        self.sec_hz = mouse.smooth_factor
        self.reward = reward
        if reward == 'water':
            self.dict_name = 'water_licks_onsets.npy'
        else:
            self.dict_name = 'food_consumption_onsets.npy'
        self.main_analyzer = MainAnalyzer(self.sec_hz)

    def single_plot_config(self, df, color, title, perc_min, perc_max):
        plt.title(title, fontsize=10)
        min_value = np.percentile(df.to_numpy(), perc_min)
        max_value = np.percentile(df.to_numpy(), perc_max)
        img = plt.imshow(df, cmap=color, norm=TwoSlopeNorm(vmin=min_value, vcenter=0, vmax=max_value),
                         aspect='auto', interpolation='nearest', rasterized=True)
        plt.colorbar(img)
        plt.axvline(x=self.sec_hz * 3, linewidth=1, color='black', linestyle='--')
        number_of_steps = np.arange(0, df.shape[1] + 1, self.sec_hz * 2)
        plt.xticks(number_of_steps, np.arange(-3, len(number_of_steps) + 1, 2))

    def visualize(self, all_days_responses):
        days_min = [13, 10]
        days_max = [85, 90]

        color1 = sns.diverging_palette(255, 10, sep=100, n=100, as_cmap=True)
        color2 = sns.diverging_palette(255, 10, sep=60, n=100, as_cmap=True)
        colors = [color1, color2]

        plt.figure(figsize=[24, 24])
        plt.suptitle(f'{self.mouse.name} - {self.reward} reward heatmaps', size=20)

        for i, day in enumerate(all_days_responses):
            plt.subplot(1, 2, i + 1)
            self.single_plot_config(day, colors[i], f'Day {i + 1}', days_min[i], days_max[i])

        plt.savefig(join(PATH, f'{self.mouse.name}_{self.reward}_stability_heatmaps.svg'))
        plt.close()

    def sort_responses(self, df):
        mean_values = df.iloc[:, round(self.sec_hz * 3): round(5 * self.sec_hz)].mean(axis=1)
        sorted_df = df.assign(mean=mean_values).sort_values('mean')
        sorted_df = sorted_df.drop('mean', axis=1)
        return sorted_df.index

    def select_mean_responses(self, cell, index, onsets):
        all_trials = self.main_analyzer.load_trials_by_index(index, cell)
        bl_sub = all_trials.sub(all_trials.iloc[:, :self.sec_hz * 2].mean(axis=1), axis=0)
        rewards_responses = pd.DataFrame(
            [bl_sub.iloc[t][onsets[t] - self.sec_hz * 3: onsets[t] + 6 * self.sec_hz].values
             for t in range(len(bl_sub))])
        return rewards_responses.mean(axis=0)

    def analyze_rewards(self, day, index_dict):
        day_data = day.load_data_dict()
        cells = self.main_analyzer.min_max_normalization(day_data['cells'])
        first_opto = np.where(day_data['cues'] == day.opto_trigger_signal)[0][0]

        if self.reward == 'water':
            index = [i for i in index_dict['drank_index'] if i < first_opto]
            onsets = index_dict['reward_drank'][:len(index)]
        else:
            index = [i for i in index_dict['ate_index'] if i < first_opto]
            onsets = index_dict['reward_ate'][:len(index)]

        with Pool() as pool:
            rewards = pool.starmap(self.select_mean_responses, [(cell, index, onsets) for cell in cells])

        return pd.DataFrame(rewards)

    def run(self):
        location_data = join(PATH, f'{self.mouse.name}_{self.reward}_heatmaps.npy')
        if not isfile(location_data):
            dict_onsets = np.load(join(self.data_dir, self.dict_name), allow_pickle=True)[()]
            responses = [self.analyze_rewards(day, dict_onsets[day.name]) for day in self.days[:2]]
            heatmap_order = self.sort_responses(responses[1])
            reordered_responses = [pd.DataFrame([response.iloc[i] for i in heatmap_order]) for response in responses]
            data = {'d1': reordered_responses[0], 'd2': reordered_responses[1]}
            np.save(location_data, data)
        else:
            data = np.load(location_data, allow_pickle=True)[()]
            reordered_responses = [data['d1'], data['d2']]

        self.visualize(reordered_responses)

        print(f'finished figures creation for {self.mouse.name}')


def main():
    for reward in ['food']:
        week_type = f'opto_{reward}_week'
        days = f'opto_{reward}_days'
        for mouse_dict in [MOUSE_YP83]:
            if week_type in mouse_dict.keys():
                mouse = Mouse(mouse_dict, mouse_dict[days], mouse_dict[week_type])
                StabilityHeatmaps(mouse, reward).run()


'__main__' == __name__ and main()
