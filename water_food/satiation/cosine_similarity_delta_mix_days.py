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
from cosine_similarity_deltas import PopulationVectorRewardsDelta

WEEK = 'week1'
DAYS = 'week1_days'
ANALYSIS_TYPE = 'reward'
PATH = join(RESULTS_PATH, 'water_food', 'satiation', 'delta_cosine_similarity', 'days_1_3')


class PopulationVectorTwoRewardsDelta:

    def __init__(self, mouse):
        self.mouse = mouse
        self.days = self.mouse.days
        self.week_data = mouse.week.data_dir_path
        self.smoothing_window = 2
        self.n_bl = 15

    def calculate_population_vector(self, responses):
        n_trials = min(30, int(len(responses[0]) /2))
        responses_smoothed = pd.DataFrame(
            [uniform_filter1d(responses[cell], size=self.smoothing_window) for cell in range(len(responses))])
        bl = responses_smoothed.iloc[:, :self.n_bl].mean(axis=1)
        results = [
            cosine_similarity(bl.to_numpy().reshape(1, -1), responses_smoothed[trial].to_numpy().reshape(1, -1))[0][0]
            for trial in range(len(responses_smoothed.iloc[0]))]
        return np.mean(results[:n_trials]) - np.mean(results[-n_trials:])

    def run(self):
        dict_responses = np.load(
            join(self.week_data, 'dict_responses_per_trial.npy'), allow_pickle=True)[()]
        responses_water = self.calculate_population_vector(dict_responses['day_1']['water_reward'])
        responses_food = self.calculate_population_vector(dict_responses['day_3']['food_reward'])
        print(f'finished data processing {self.mouse.name}')
        return responses_water, responses_food


def mean_mice_delta(water, food, only_water, only_food):
    p_value_water = round(ttest_ind(water, only_water)[1], 4)
    p_value_food = round(ttest_ind(food, only_food)[1], 4)

    plt.figure(figsize=[18, 18])
    plt.suptitle(f'Summary deltas water: {p_value_water}, food: {p_value_food}', fontsize=24)
    plt.bar(
        ['water', 'water_only', 'food', 'food_only'],
        [np.mean(water), np.mean(only_water), np.mean(food), np.mean(only_food)],
        color=['royalblue', 'darkblue', 'lightcoral', 'darkred'])

    for m in water:
        plt.scatter('water', m, color='black', alpha=0.5)
    for m in food:
        plt.scatter('food', m, color='black', alpha=0.5)
    for m in only_water:
        plt.scatter('water_only', m, color='black', alpha=0.5)
    for m in only_food:
        plt.scatter('food_only', m, color='black', alpha=0.5)
    plt.ylabel('Cosine similarity delta', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.savefig(join(PATH, f'summary_cosine_similarity_deltas_days.svg'))
    plt.close()


def main():
    water_delta = []
    food_delta = []
    only_water = []
    only_food = []
    for mouse_dict in ALL_MICE:
        if WEEK in mouse_dict.keys():
            mouse = Mouse(mouse_dict, mouse_dict[DAYS], mouse_dict[WEEK])
            processor = PopulationVectorTwoRewardsDelta(mouse)
            water, food = processor.run()
            water_delta.append(water)
            food_delta.append(food)

    for mouse_dict in ALL_MICE:
        if 'opto_water_week' in mouse_dict.keys():
            mouse = Mouse(mouse_dict, mouse_dict['opto_water_days'], mouse_dict['opto_water_week'])
            only_water.append(PopulationVectorRewardsDelta(mouse, 'water').run())

        if 'opto_food_week' in mouse_dict.keys():
            mouse = Mouse(mouse_dict, mouse_dict['opto_food_days'], mouse_dict['opto_food_week'])
            only_food.append(PopulationVectorRewardsDelta(mouse, 'food').run())

    mean_mice_delta(water_delta, food_delta, only_water, only_food)


'__main__' == __name__ and main()
