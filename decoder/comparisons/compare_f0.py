import numpy as np
import pandas as pd
from multiprocessing import Pool
from os.path import join, isdir
import matplotlib.pyplot as plt

from configurations import *
from run import Run
from mouse import Mouse
from Processor import Processor
from main_analyzer import MainAnalyzer

RELEVANT_DAYS = ['opto_water_days']
RELEVANT_MICE = [MOUSE_YP84]
N_DFF = 10


class Normalization(Processor):

    def __init__(self, day, *args, **kwargs):
        Processor.__init__(self, *args, **kwargs)
        self.day = day
        self.sec_hz = self.day.sec_hz
        self.chunk = self.sec_hz * 8
        self.window_iti = int(self.chunk * N_DFF)
        self.window_whole = int(self.sec_hz * 60 * 4.5)
        self.frames_per_run = day.find_frames_per_run()
        self.main_analyzer = MainAnalyzer(self.sec_hz)

    def normalization(self, cell, window):
        f0 = cell.rolling(window=window, min_periods=window).quantile(0.1)
        first_window = np.percentile(cell.iloc[0: window], 10)
        f0[:window] = first_window
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
        return good_neutral_trials_index, day_trials

    def visualize(self, whole_trace, iti_trace, whole_fo, iti_f0, no_norm):
        for cell in range(len(whole_trace[0])):
            cell_whole = pd.concat([run.iloc[cell] for run in whole_trace], ignore_index=True)
            cell_iti = pd.concat([run.iloc[cell] for run in iti_trace], ignore_index=True)
            cell_whole_f0 = pd.concat([run.iloc[cell] for run in whole_fo], ignore_index=True)
            cell_iti_f0 = pd.concat([run.iloc[cell] for run in iti_f0], ignore_index=True)
            cell_no_norm = pd.concat([run.iloc[cell] for run in no_norm], ignore_index=True)

            plt.figure(figsize=(20, 30))
            plt.subplot(5, 1, 1)
            plt.title(f'Cell {cell} whole trace')
            plt.plot(cell_whole, color='blue')
            plt.subplot(5, 1, 2)
            plt.title(f'Cell {cell} whole f0 trace')
            plt.plot(cell_whole_f0, color='red')
            plt.subplot(5, 1, 3)
            plt.title(f'Cell {cell} ITI trace')
            plt.plot(cell_iti, color='blue')
            plt.subplot(5, 1, 4)
            plt.title(f'Cell {cell} ITI f0 trace')
            plt.plot(cell_iti_f0, color='red')
            plt.subplot(5, 1, 5)
            plt.title(f'Cell {cell} no norm trace')
            plt.plot(cell_no_norm, color='blue')

            plt.savefig(join(RESULTS_PATH, 'cell_f0', f'{self.day.mouse_name}_{self.day.name}_cell{cell}.jpg'))
            plt.close()

    def run(self):
        print('processing..')
        relevant_trials, trials_type = self.find_relevant_trials()
        day_activity = np.load(
            join(self.day.data_dir_path, 'clean_movement_artifacts', 'clean_data.npy'), allow_pickle=True)[:50]
        data_dicts, _ = self.day.load_inputs_dict_per_run()
        trials_onsets = [d['trials_onsets'] for d in data_dicts]
        by_run_relevant_trials = self.find_by_run_relevant_trials(trials_onsets, relevant_trials)
        if len(self.frames_per_run) - 1 > len(trials_onsets):
            self.frames_per_run = self.frames_per_run[:len(trials_onsets) + 1]
            print(f'{self.day.mouse_name} {self.day.name} - frames per run was fixed')

        with Pool() as pool:
            day_data_itis = []
            day_data = []
            day_f0_itis = []
            day_f0 = []
            no_normalization = []
            for run_i in range(len(self.frames_per_run) - 1):
                run_onsets = trials_onsets[run_i]
                run_relevant_trials = by_run_relevant_trials[run_i]
                start_run = self.frames_per_run[run_i]
                end_run = self.frames_per_run[run_i + 1]
                run = day_activity[:, start_run: end_run]
                no_normalization.append(pd.DataFrame(run))

                f0 = pd.DataFrame(pool.starmap(self.normalization,
                                               [(pd.Series(cell), self.window_whole) for cell in run]))
                normalized_data = (pd.DataFrame(run) - f0) / f0
                day_data.append(normalized_data)
                day_f0.append(f0)

                cells_itis = pool.starmap(self.cells_itis, [(cell, run_onsets, run_relevant_trials) for cell in run])
                f0_itis = pd.DataFrame(pool.starmap(self.normalization,
                                       [(pd.Series(cell), self.window_iti) for cell in cells_itis]))
                normalized_data_itis = (pd.DataFrame(cells_itis) - f0_itis) / f0_itis
                day_data_itis.append(normalized_data_itis)
                day_f0_itis.append(f0_itis)
            print('creating figures')
            self.visualize(day_data, day_data_itis, day_f0, day_f0_itis, no_normalization)


def main(mice_paths, relevant_days):
    for m in mice_paths:
        for days in relevant_days:
            if days in m.keys():
                mouse = Mouse(m, m[days])
                if 'opto' in days:
                    days_to_analyze = mouse.days
                else:
                    days_to_analyze = mouse.days[:5]
                for day in days_to_analyze:
                    process = Normalization(day)
                    process.run()


'__main__' == __name__ and main(RELEVANT_MICE, RELEVANT_DAYS)
