import numpy as np
from os.path import join
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

from configurations import *
from mouse import Mouse
from main_analyzer import MainAnalyzer
from Visualizer import Visualizer

PATH = join(RESULTS_PATH, 'water_food', 'stability', 'correlations')


class ModulationIndexBetween:

    def __init__(self, mouse, reward):
        self.mouse = mouse
        self.days = [d for d in self.mouse.days if d.name[-1] != 'b']
        self.week = self.mouse.week
        self.sec_hz = mouse.smooth_factor
        self.reward = reward
        self.main_analyzer = MainAnalyzer(self.sec_hz)
        self.visualizer = Visualizer(self.sec_hz)
        if reward == 'water':
            self.dict_name = 'water_licks_onsets.npy'
        else:
            self.dict_name = 'food_consumption_onsets.npy'

    def visualize(self, day1, day2, day3):
        plt.figure(figsize=(21, 21))
        plt.suptitle(f'{self.mouse.name}', size=30)

        plt.subplot(2, 1, 1)
        self.visualizer.scatter_plot_config([day2, day1], ['day2', 'day1'], 'days 1-2')

        plt.subplot(2, 1, 2)
        self.visualizer.scatter_plot_config([day2, day3], ['day2', 'day3'], 'days 2-3')
        plt.savefig(join(PATH, f'{self.reward}_{self.mouse.name}.svg'))
        plt.close()

    def analyze_day(self, day, onsets, responses):
        first_opto = np.where(day.load_data_dict()['cues'] == day.opto_trigger_signal)[0][0]
        if self.reward == 'water':
            index_reward = len([i for i in onsets['drank_index'] if i < first_opto])
        else:
            index_reward = len([i for i in onsets['ate_index'] if i < first_opto])
        cells_responses = [np.mean(cell[:index_reward]) for cell in responses]
        return cells_responses

    def run(self):
        dict_onsets = np.load(join(self.week.data_dir_path, self.dict_name), allow_pickle=True)[()]
        dict_responses = np.load(join(self.week.data_dir_path, 'dict_mean_responses_per_trial.npy'), allow_pickle=True)[()]

        day1, day2, day3 = [
            self.analyze_day(day, dict_onsets[day.name], dict_responses[f'day{i + 1}_reward'])
            for i, day in enumerate(self.days)]

        self.visualize(day1, day2, day3)

        return [np.corrcoef(day1, day2)[0, 1], np.corrcoef(day2, day3)[0, 1]]


def summary_mice(correlations, reward):
    x_labels = ['days 1-2', 'days 2-3']
    mean_1_2 = np.mean([m[0] for m in correlations])
    mean_2_3 = np.mean([m[1] for m in correlations])

    plt.figure(figsize=(12, 12))
    plt.title(f'Stability correlations {reward}', size=30)

    plt.bar(x_labels, [mean_1_2, mean_2_3])
    for mouse in correlations:
        plt.plot(x_labels, mouse, color='gray', alpha=0.5)
    plt.yticks(size=15)
    plt.xticks(size=15)

    plt.savefig(join(PATH, f'summary_{reward}_stability.svg'))
    plt.close()


def main():
    for reward in ['food']:
        all_correlations = []
        week_type = f'opto_{reward}_week'
        days = f'opto_{reward}_days'
        for mouse_dict in [MOUSE_YP83]:
            if week_type in mouse_dict.keys():
                mouse = Mouse(mouse_dict, mouse_dict[days], mouse_dict[week_type])
                all_correlations.append(ModulationIndexBetween(mouse, reward).run())
                print(f'finished {mouse.name} {reward}')

        # summary_mice(all_correlations, reward)


'__main__' == __name__ and main()

