import numpy as np
import pandas as pd
from os.path import join
from multiprocessing import Pool

from configurations import *
from mouse import Mouse
from main_analyzer import MainAnalyzer

MICE = ALL_MICE
WEEK_TYPE = 'week1'
DAYS = 'week1_days'
MIN_TRIALS = 30


class DictMeanResponses:

    def __init__(self, mouse, min_trials):
        self.mouse = mouse
        self.days = self.mouse.days
        self.sec_hz = mouse.smooth_factor
        self.min_trials = min_trials
        self.week_data = mouse.week.data_dir_path
        self.main_analyzer = MainAnalyzer(self.sec_hz)
        self.dict_onsets = join(mouse.week.data_dir_path, 'rewards_indexes_onsets.npy')

    def save_data(self, days_responses):
        dict_classification = {'order': 'cue, reward'}

        for day in range(len(days_responses)):
            water, food = days_responses[day]
            dict_classification[f'day_{day + 1}'] = {
                'water_cue': water[0],
                'water_reward': water[1],
                'food_cue': food[0],
                'food_reward': food[1]
            }

        np.save(join(self.week_data, 'dict_mean_responses.npy'), dict_classification)

    def select_mean_responses(self, cell, index, onsets):
        all_trials = self.main_analyzer.load_trials_by_index(index, cell)
        bl_sub = all_trials.sub(all_trials.iloc[:, :self.sec_hz * 2].mean(axis=1), axis=0)
        rewards_responses = pd.DataFrame([bl_sub.iloc[t][onsets[t]: onsets[t] + 2 * self.sec_hz].values
                                          for t in range(len(bl_sub))])
        return rewards_responses.mean(axis=0).mean()

    def analyze_rewards(self, day, index_dict):
        day_data = day.load_data_dict()
        cells = self.main_analyzer.min_max_normalization(day_data['cells'])

        water_rewards, water_cues, food_rewards, food_cues = [], [], [], []

        with Pool() as pool:
            if len(index_dict['index_drank']) > 10:
                index = index_dict['index_drank'][:self.min_trials]
                onset_cues = [int(self.sec_hz * 4)] * len(index)
                onsets_reward = index_dict['onsets_drank'][:self.min_trials]
                water_rewards = pool.starmap(self.select_mean_responses,
                                             [(cell, index, onsets_reward) for cell in cells])
                water_cues = pool.starmap(self.select_mean_responses, [(cell, index, onset_cues) for cell in cells])

            if len(index_dict['index_ate_full']) > 10:
                index = index_dict['index_ate_full'][:self.min_trials]
                onset_cues = [int(self.sec_hz * 5)] * len(index)
                onsets_reward = index_dict['onsets_ate_full'][:self.min_trials]
                food_rewards = pool.starmap(self.select_mean_responses,
                                            [(cell, index, onsets_reward) for cell in cells])
                food_cues = pool.starmap(self.select_mean_responses, [(cell, index, onset_cues) for cell in cells])

        return [water_cues, water_rewards], [food_cues, food_rewards]

    def run(self):
        dict_responses = np.load(self.dict_onsets, allow_pickle=True)[()]
        days_responses = []
        for day_i, day in enumerate(self.days[:4]):
            days_responses.append(self.analyze_rewards(day, dict_responses[f'day{day_i + 1}']))
            print(f'finished day {day.name}')

        self.save_data(days_responses)

        print(f'finished data processing {self.mouse.name}')


def main():
    for mouse_dict in MICE:
        if WEEK_TYPE in mouse_dict.keys():
            mouse = Mouse(mouse_dict, mouse_dict[DAYS], mouse_dict[WEEK_TYPE])
            processor = DictMeanResponses(mouse, MIN_TRIALS)
            processor.run()


'__main__' == __name__ and main()
