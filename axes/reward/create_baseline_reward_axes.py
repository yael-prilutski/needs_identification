import numpy as np
from os.path import join

from configurations import *
from mouse import Mouse
from Processor import Processor
from axes.axes_analyzer import AxesAnalyzer
from main_analyzer import MainAnalyzer

RELEVANT_DAYS = 'week1_days'
WEEK = 'week1'
RELEVANT_MICE = ALL_MICE
N_TRIALS = 30


class WithinDayAxis(Processor):

    def __init__(self, mouse, vector_name, *args, **kwargs):
        Processor.__init__(self, *args, **kwargs)
        self.days = mouse.days
        self.vector_name = vector_name
        self.sec_hz = mouse.smooth_factor
        self.week = mouse.week
        self.analyzer = AxesAnalyzer(mouse.smooth_factor)
        self.main_analyzer = MainAnalyzer(mouse.smooth_factor)

    def select_cue_response(self, day, trial_type):
        day_data = day.load_data_dict()
        normalized_cells = self.main_analyzer.min_max_normalization(day_data['cells'])
        if type(trial_type) is not list:
            relevant_trials = np.where(day_data['trials_classification'] == trial_type)[0][:N_TRIALS]
        else:
            relevant_trials = np.where((day_data['trials_classification'] == trial_type[0]) |
                                       (day_data['trials_classification'] == trial_type[1]))[0][:N_TRIALS]

        baseline_responses = []
        reward_responses = []
        for cell in normalized_cells:
            baseline_trials = np.mean([cell[t][:self.sec_hz * 2].mean() for t in relevant_trials])
            reward_trials = np.mean([cell[t][self.sec_hz * 4: self.sec_hz * 6].mean() for t in relevant_trials])
            baseline_responses.append(baseline_trials)
            reward_responses.append(reward_trials)

        sub_baseline_reward = np.array(baseline_responses) - np.array(reward_responses)
        dict_axis = {
            'sub': sub_baseline_reward,
            'start': np.dot(reward_responses, sub_baseline_reward),
            'end': np.dot(baseline_responses, sub_baseline_reward)}
        return dict_axis

    def run(self):
        vectors_location = join(self.week.data_dir_path, 'axes', self.vector_name)
        axes_dict = {
            'water': self.select_cue_response(self.days[1], self.days[1].drank),
            'food': self.select_cue_response(self.days[3], [self.days[3].ate, self.omission_food_taste])}
        np.save(vectors_location, axes_dict)


def main():
    for mouse_path in RELEVANT_MICE:
        if WEEK in mouse_path.keys():
            mouse = Mouse(mouse_path, mouse_path[RELEVANT_DAYS], mouse_path[WEEK])
            process = WithinDayAxis(mouse, vector_name='reward_baseline_axes.npy')
            process.run()
            print(f'finished processing {mouse.name}')


'__main__' == __name__ and main()
