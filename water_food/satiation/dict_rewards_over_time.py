import numpy as np
from os.path import join
from multiprocessing import Pool

from configurations import *
from mouse import Mouse
from main_analyzer import MainAnalyzer

MICE = ALL_MICE


class DictMeanResponses:

    def __init__(self, mouse, reward):
        self.mouse = mouse
        self.days = self.mouse.days
        self.sec_hz = mouse.smooth_factor
        self.reward = reward
        self.week_data = mouse.week.data_dir_path
        self.main_analyzer = MainAnalyzer(self.sec_hz)
        if reward == 'water':
            self.dict_onsets = join(mouse.week.data_dir_path, 'water_licks_onsets.npy')
        else:
            self.dict_onsets = join(mouse.week.data_dir_path, 'food_consumption_onsets.npy')

    def save_data(self, day1, day2, day3):
        dict_classification = {
            'day1_cue': day1[0],
            'day1_reward': day1[1],
            'day2_cue': day2[0],
            'day2_reward': day2[1],
            'day3_cue': day3[0],
            'day3_reward': day3[1]
        }

        np.save(join(self.week_data, 'dict_mean_responses_per_trial.npy'), dict_classification)

    def select_mean_responses(self, cell, index, start, end):
        all_trials = self.main_analyzer.load_trials_by_index(index, cell)
        bl_sub = all_trials.sub(all_trials.iloc[:, :self.sec_hz * 2].mean(axis=1), axis=0)
        mean_rewards = [bl_sub.iloc[t][start[t]: end[t]].mean() for t in range(len(bl_sub))]
        return mean_rewards

    def analyze_rewards(self, day, index_dict):
        day_data = day.load_data_dict()
        cells = self.main_analyzer.min_max_normalization(day_data['cells'])

        if self.reward == 'water':
            index = index_dict['drank_index']
            onsets_reward = index_dict['reward_drank']
        else:
            index = index_dict['ate_index']
            onsets_reward = index_dict['reward_ate']

        onset_cues = [int(self.sec_hz * 4)] * len(index)
        end_cues = [int(self.sec_hz * 6)] * len(index)
        end_reward = [o + int(self.sec_hz * 2) for o in onsets_reward]
        with Pool() as pool:
            responses_rewards = pool.starmap(self.select_mean_responses,
                                             [(cell, index, onsets_reward, end_reward) for cell in cells])
            responses_cues = pool.starmap(self.select_mean_responses,
                                          [(cell, index, onset_cues, end_cues) for cell in cells])

        return responses_cues, responses_rewards

    def run(self):
        dict_responses = np.load(self.dict_onsets, allow_pickle=True)[()]
        days = [d for d in self.days if d.name[-1] != 'b']
        day1 = self.analyze_rewards(days[0], dict_responses[days[0].name])
        day2 = self.analyze_rewards(days[1], dict_responses[days[1].name])
        day3 = self.analyze_rewards(days[2], dict_responses[days[2].name])
        self.save_data(day1, day2, day3)

        print(f'finished data processing {self.mouse.name}')


def main():
    for reward in ['water', 'food']:
        days = f'opto_{reward}_days'
        week = f'opto_{reward}_week'
        for mouse_dict in MICE:
            if week in mouse_dict.keys():
                mouse = Mouse(mouse_dict, mouse_dict[days], mouse_dict[week])
                processor = DictMeanResponses(mouse, reward)
                processor.run()


'__main__' == __name__ and main()
