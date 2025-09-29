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

    def find_trials_onsets(self, relevant_trials, licks_water, licks_food):
        onsets_for_trial = []
        for trial in relevant_trials:
            stop_data = self.sec_hz * 4
            first_lick_food, first_lick_water = [], []
            if len(licks_food):
                first_lick_food = [i for i in np.where(np.array(licks_food[trial]) > 0)[0] if i > self.sec_hz * 2]
            if len(licks_water):
                first_lick_water = [i for i in np.where(np.array(licks_water[trial]) > 0)[0] if i > self.sec_hz * 2]

            if len(first_lick_water) > 0:
                onset_lick = first_lick_water[0] - int(self.sec_hz / 10)
                if onset_lick < stop_data:
                    stop_data = onset_lick
            if len(first_lick_food) > 0:
                onset_lick = first_lick_food[0] - int(self.sec_hz / 10)
                if onset_lick < stop_data:
                    stop_data = onset_lick
            onsets_for_trial.append(stop_data)

        good_trials = [i for i in range(len(onsets_for_trial)) if onsets_for_trial[i] > self.sec_hz * 2.5][:N_TRIALS]
        final_trials = [relevant_trials[i] for i in good_trials]
        final_onsets = [onsets_for_trial[i] for i in good_trials]
        return final_trials, final_onsets

    def select_cue_response(self, day, trial_type):
        day_data = day.load_data_dict()
        normalized_cells = self.main_analyzer.min_max_normalization(day_data['cells'])
        licks_water = day_data['water_licks']
        licks_food = day_data['pellet_licks']
        relevant_trials = np.where(day_data['cues'] == trial_type)[0]
        final_trials, trials_onsets = self.find_trials_onsets(relevant_trials, licks_water, licks_food)

        baseline_responses = []
        cue_responses = []
        for cell in normalized_cells:
            baseline_trials = np.mean([cell[t][:self.sec_hz * 2].mean() for t in final_trials])
            cue_trials = np.mean(
                [cell[t][self.sec_hz * 2: trials_onsets[i]].mean() for i, t in enumerate(final_trials)])
            baseline_responses.append(baseline_trials)
            cue_responses.append(cue_trials)

        sub_baseline_cue = np.array(baseline_responses) - np.array(cue_responses)
        dict_axis = {
            'sub': sub_baseline_cue,
            'start': np.dot(cue_responses, sub_baseline_cue),
            'end': np.dot(baseline_responses, sub_baseline_cue)}
        return dict_axis

    def run(self):
        vectors_location = join(self.week.data_dir_path, 'axes', self.vector_name)
        axes_dict = {
            'water': self.select_cue_response(self.days[1], self.days[1].fluid_signal),
            'neutral_water': self.select_cue_response(self.days[1], self.days[1].neutral_signal),
            'food': self.select_cue_response(self.days[3], self.days[3].pellet_signal),
            'neutral_food': self.select_cue_response(self.days[3], self.days[3].neutral_signal)}
        np.save(vectors_location, axes_dict)


def main():
    for mouse_path in RELEVANT_MICE:
        if WEEK in mouse_path.keys():
            mouse = Mouse(mouse_path, mouse_path[RELEVANT_DAYS], mouse_path[WEEK])
            process = WithinDayAxis(mouse, vector_name='cue_baseline_axes.npy')
            process.run()
            print(f'finished processing {mouse.name}')


'__main__' == __name__ and main()
