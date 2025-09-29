import numpy as np
import pandas as pd
from os.path import join
from multiprocessing import Pool

from configurations import *
from mouse import Mouse
from main_analyzer import MainAnalyzer

MICE = ALL_MICE
MIN_TRIALS = 30


class DictMeanResponses:

    def __init__(self, mouse, min_trials, reward):
        self.mouse = mouse
        self.days = self.mouse.days
        self.sec_hz = mouse.smooth_factor
        self.min_trials = min_trials
        self.reward = reward
        self.week_data = mouse.week.data_dir_path
        self.main_analyzer = MainAnalyzer(self.sec_hz)
        if reward == 'water':
            self.dict_name = 'water_licks_onsets.npy'
        else:
            self.dict_name = 'food_consumption_onsets.npy'

    def save_data(self, days_responses):
        dict_classification = {}
        for day in range(len(days_responses)):
            dict_classification[f'day_{day + 1}'] = {
                'reward': days_responses[day][0],
                'cue': days_responses[day][1]
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

        if self.reward == 'water':
            index = index_dict['drank_index'][:self.min_trials]
            onset_cues = [int(self.sec_hz * 4)] * len(index)
            onsets_reward = index_dict['reward_drank'][:self.min_trials]
        else:
            index = index_dict['ate_index'][:self.min_trials]
            onset_cues = [int(self.sec_hz * 4)] * len(index)
            onsets_reward = index_dict['reward_ate'][:self.min_trials]

        with Pool() as pool:
            rewards = pool.starmap(self.select_mean_responses, [(cell, index, onsets_reward) for cell in cells])
            cues = pool.starmap(self.select_mean_responses, [(cell, index, onset_cues) for cell in cells])

        return rewards, cues

    def run(self):
        dict_responses = np.load(join(self.week_data, self.dict_name), allow_pickle=True)[()]
        relevant_days = [day for day in self.days if day.name[-1] != 'b'][:2]
        days_responses = [self.analyze_rewards(day, dict_responses[day.name]) for day in relevant_days]
        self.save_data(days_responses)

        print(f'finished data processing {self.mouse.name}')


def main():
    for reward in ['food']:
        week_type = f'opto_{reward}_week'
        days = f'opto_{reward}_days'
        for mouse_dict in MICE:
            if week_type in mouse_dict.keys():
                mouse = Mouse(mouse_dict, mouse_dict[days], mouse_dict[week_type])
                processor = DictMeanResponses(mouse, MIN_TRIALS, reward)
                processor.run()


'__main__' == __name__ and main()
