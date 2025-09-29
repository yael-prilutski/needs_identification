import numpy as np
import pandas as pd
from multiprocessing import Pool
from os.path import join, isdir

from configurations import *
from run import Run
from mouse import Mouse
from Processor import Processor
from main_analyzer import MainAnalyzer

# RELEVANT_DAYS = [['week1_days', 'week1'], ['week2_days', 'week2'], ['opto_water_days', 'opto_water_week'],
#                  ['opto_food_days', 'opto_food_week']]
RELEVANT_DAYS = [['opto_water_days', 'opto_water_week']]
RELEVANT_MICE = [MOUSE_YP82]
N_DFF = 10


class Normalization(Processor):

    def __init__(self, day, *args, **kwargs):
        Processor.__init__(self, *args, **kwargs)
        self.day = day
        self.sec_hz = self.day.sec_hz
        self.chunk = self.sec_hz * 8
        self.window_fr = int(self.chunk * N_DFF)
        self.frames_per_run = day.find_frames_per_run()
        self.main_analyzer = MainAnalyzer(self.sec_hz)

    def _save_file(self, day_data, relevant_trials, trials_type):
        data_dict = {
            'relevant_trials': relevant_trials,
            'trials_type': trials_type
        }
        for run_i, run in enumerate(day_data):
            data_dict[f'run{run_i + 1}'] = run

        file_name = f'{self.day.mouse_name}_{self.day.name}_itis_sw_dff.npy'
        file_location = join(self.day.data_dir_path, file_name)
        np.save(file_location, data_dict)
        print(f'saved {self.day.mouse_name} {self.day.name}')

    def normalization(self, cell):
        f0 = cell.rolling(window=self.window_fr, min_periods=self.window_fr).quantile(0.1)
        first_window = np.percentile(cell.iloc[0: self.window_fr], 10)
        f0[:self.window_fr] = first_window
        return f0.to_numpy()

    def cells_itis(self, cell, trials_onsets, run_relevant_trials):
        real_trials_onsets = [o + self.sec_hz * 2 for o in trials_onsets]
        cell_by_trials = np.split(np.array(cell), real_trials_onsets, axis=0)[1:-1]
        cell_itis = np.concatenate([t[-self.chunk:] for i, t in enumerate(cell_by_trials) if i in run_relevant_trials])
        return pd.Series(cell_itis)

    def find_by_run_relevant_trials(self, trials_onsets, relevant_trials):
        len_trials_per_run = [len(onsets) - 1 for onsets in trials_onsets]
        n_trials = 0
        relevant_trials_per_run = []
        for run_i in range(len(len_trials_per_run)):
            real_trials_indexes = np.arange(len_trials_per_run[run_i]) + n_trials
            run_relevant_trials = [i for i, t in enumerate(real_trials_indexes) if t in relevant_trials]
            relevant_trials_per_run.append(run_relevant_trials)
            n_trials += len_trials_per_run[run_i]
        return relevant_trials_per_run

    def find_relevant_trials(self):
        day_data = self.day.load_data_dict()
        day_trials = day_data['trials_classification']
        licks_water = day_data['water_licks']
        licks_food = day_data['pellet_licks']
        if self.day.is_light:
            day_trials = self.main_analyzer.change_to_regular_opto_trials(self.day, day_trials)
        ignore_pellet_licks = sum([Run(path, self.sec_hz).constant_pellet_licks for path in self.day.runs_paths])
        neutral_index = np.where(day_trials == self.day.neutral)[0]
        good_neutral_trials_index = []
        for i in neutral_index:
            try:
                water_licks = licks_water[i][-10 * self.sec_hz:]
            except IndexError:
                water_licks = [0]
            food_licks = [0]
            if not ignore_pellet_licks:
                try:
                    food_licks = licks_food[i][-10 * self.sec_hz:]
                except IndexError:
                    pass
            if sum(water_licks) == 0 and sum(food_licks) == 0:
                good_neutral_trials_index.append(i)
        # print(f'Perc good trials: {round(len(good_neutral_trials_index) / len(neutral_index) * 100, 1)}%')
        return good_neutral_trials_index, day_trials

    def run(self, iti_dict):
        print('processing..')
        # relevant_trials = iti_dict['relevant_trials']
        # trials_type = iti_dict['trials_classification']
        relevant_trials, trials_type = self.find_relevant_trials()
        day_activity = np.load(
            join(self.day.data_dir_path, 'clean_movement_artifacts', 'clean_data.npy'), allow_pickle=True)
        data_dicts, _ = self.day.load_inputs_dict_per_run()
        trials_onsets = [d['trials_onsets'] for d in data_dicts]
        by_run_relevant_trials = self.find_by_run_relevant_trials(trials_onsets, relevant_trials)
        if len(self.frames_per_run) - 1 > len(trials_onsets):
            self.frames_per_run = self.frames_per_run[:len(trials_onsets) + 1]
            print(f'{self.day.mouse_name} {self.day.name} - frames per run was fixed')

        with Pool() as pool:
            day_data = []
            for run_i in range(len(self.frames_per_run) - 1):
                run_onsets = trials_onsets[run_i]
                run_relevant_trials = by_run_relevant_trials[run_i]
                start_run = self.frames_per_run[run_i]
                end_run = self.frames_per_run[run_i + 1]
                run = day_activity[:, start_run: end_run]

                cells_itis = pool.starmap(self.cells_itis, [(cell, run_onsets, run_relevant_trials) for cell in run])
                f0 = pd.DataFrame(pool.map(self.normalization, cells_itis))
                normalized_data = (pd.DataFrame(cells_itis) - f0) / f0
                day_data.append(normalized_data)
            self._save_file(day_data, relevant_trials, trials_type)


def main(mice_paths, relevant_days):
    for m in mice_paths:
        for days, week in relevant_days:
            if days in m.keys():
                mouse = Mouse(m, m[days], m[week])
                iti_dict = np.load(
                    join(mouse.week.data_dir_path, 'axes', f'{mouse.name}_iti_dict.npy'), allow_pickle=True)[()]
                if 'opto' in days:
                    days_to_analyze = mouse.days
                else:
                    days_to_analyze = mouse.days[:5]
                for day in days_to_analyze:
                    process = Normalization(day)
                    process.run(iti_dict[day.name])


'__main__' == __name__ and main(RELEVANT_MICE, RELEVANT_DAYS)
