import numpy as np
import pandas as pd
from os.path import join
from scipy.stats import ttest_ind
from scipy.ndimage import uniform_filter1d
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams['svg.fonttype'] = 'none'

from configurations import *
from mouse import Mouse

ANALYSIS_TYPE = 'reward'
PATH = join(RESULTS_PATH, 'water_food', 'satiation', 'delta_cosine_similarity')


class PopulationVectorRewardsDelta:

    def __init__(self, mouse, reward):
        self.mouse = mouse
        self.days = self.mouse.days
        self.week_data = mouse.week.data_dir_path
        self.reward = reward
        self.smoothing_window = 2
        self.n_bl = 15

    def calculate_population_vector(self, responses):
        responses_smoothed = pd.DataFrame(
            [uniform_filter1d(responses.iloc[cell], size=self.smoothing_window) for cell in range(len(responses))])
        bl = responses_smoothed.iloc[:, :self.n_bl].mean(axis=1)
        results = [
            cosine_similarity(bl.to_numpy().reshape(1, -1), responses_smoothed[trial].to_numpy().reshape(1, -1))[0][0]
            for trial in range(len(responses_smoothed.iloc[0]))]
        return results

    def run(self):
        dict_responses = np.load(
            join(self.week_data, 'dict_mean_responses_per_trial.npy'), allow_pickle=True)[()]
        responses = self.calculate_population_vector(pd.DataFrame(dict_responses[f'day1_{ANALYSIS_TYPE}']))
        delta = np.mean(responses[:30]) - np.mean(responses[-30:])
        print(f'finished data processing {self.mouse.name}')
        return delta


def mean_mice_delta(all_deltas):
    water, food = all_deltas
    p_value = round(ttest_ind(water, food)[1], 3)
    plt.figure(figsize=[18, 18])
    plt.suptitle(f'All mice deltas cosine similarity: {p_value}', fontsize=24)
    plt.bar(['water', 'food'], [np.mean(water), np.mean(food)], color=['royalblue', 'lightcoral'])

    for m in water:
        plt.scatter('water', m, color='black', alpha=0.5)
    for m in food:
        plt.scatter('food', m, color='black', alpha=0.5)
    plt.ylabel('Cosine similarity delta', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.savefig(join(PATH, f'all_mice_cosine_similarity_deltas_by_{ANALYSIS_TYPE}_30.svg'))
    plt.close()


def main():
    reward_deltas = []
    for reward in ['water', 'food']:
        week_type = f'opto_{reward}_week'
        days = f'opto_{reward}_days'
        all_deltas = []
        for mouse_dict in ALL_MICE:
            if week_type in mouse_dict.keys():
                mouse = Mouse(mouse_dict, mouse_dict[days], mouse_dict[week_type])
                processor = PopulationVectorRewardsDelta(mouse, reward)
                all_deltas.append(processor.run())
        reward_deltas.append(all_deltas)
    mean_mice_delta(reward_deltas)


'__main__' == __name__ and main()
