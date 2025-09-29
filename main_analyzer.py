import sys
import cv2
import numpy as np
import pandas as pd
from os import listdir
from scipy.stats import zscore
from os.path import join, isfile
from multiprocessing import Pool

from Processor import Processor
from run import Run

THRESH_MI = 0.7


class MainAnalyzer(Processor):

    def __init__(self, sec_hz, *args, **kwargs):
        Processor.__init__(self, *args, **kwargs)

        self.sec_hz = sec_hz
        self.frames_pre_trial = int(self.sec_hz * self.sec_pre_trial)
        self.cue_frames = int(self.sec_hz * self.cue_duration)
        self.thresh_mi = THRESH_MI

    @staticmethod
    def find_days(mouse_path, days_list):
        all_files = listdir(mouse_path)
        return [join(mouse_path, file) for file in all_files if len([True for day in days_list if day in file]) > 0]

    @staticmethod
    def load_trials_by_index(index, cell_data):
        if len(index) > 0:
            return pd.DataFrame([cell_data[i] for i in index])
        else:
            return []

    def min_max_single_cell(self, cell_data):
        cell_activity = np.hstack(cell_data)
        max_percentage_value = np.percentile(cell_activity, 99)
        min_percentage_value = np.percentile(cell_activity, 1)
        normalized_cell_activity = [(trial - min_percentage_value) / (max_percentage_value - min_percentage_value)
                                    for trial in cell_data]
        return normalized_cell_activity

    def min_max_normalization(self, cells_data):
        with Pool() as pool:
            return pool.map(self.min_max_single_cell, cells_data)

    def zscore_single_cell(self, cell_data):
        cell_activity = np.hstack(cell_data)
        mean_value = np.mean(cell_activity)
        std_value = np.std(cell_activity)
        normalized_cell_activity = [(trial - mean_value) / std_value for trial in cell_data]
        return normalized_cell_activity

    def zscore_normalization(self, cells_data):
        with Pool() as pool:
            return pool.map(self.zscore_single_cell, cells_data)

    def change_to_regular_opto_cues(self, day, cues):
        adapted_cues = []
        for cue in cues:
            if cue == day.fluid_light_signal:
                adapted_cues.append(day.fluid_signal)
            elif cue in [day.pellet_light_signal, day.pellet_light_off_signal, day.pellet_light_O_signal]:
                adapted_cues.append(day.pellet_signal)
            elif cue in [day.neutral_light_signal, day.neutral_light_off_signal, day.neutral_light_omission_signal]:
                adapted_cues.append(day.neutral_signal)
            else:
                adapted_cues.append(cue)
        return np.array(adapted_cues)

    def fix_opto_trials_classification(self, day, cues):
        opto_triggers = np.where(cues == day.opto_trigger)[0]
        start_light = opto_triggers[0]
        end_light = opto_triggers[-1]
        adapted_cues = cues.copy()

        for i in range(start_light, end_light):
            cue = cues[i]
            light_on = len([o for o in opto_triggers if o < i]) % 2 == 1
            for no_light_trial, light_on_trial, between_light_trial in [
                [day.drank, day.drank_light, day.drank_between_light],
                [day.not_drank, day.not_drank_light, day.not_drank_between_light],
                [day.omission_water, day.omission_fluid_light, day.omission_fluid_between_light],
                [day.neutral, day.neutral_light, day.neutral_between_light],
                [day.ate, day.ate_light, day.ate_between_light],
                [day.not_ate, day.not_ate_light, day.not_ate_between_light],
                [day.not_ate_licked, day.not_ate_light, day.not_ate_between_light],
                [day.omission_food_taste, day.ate_light, day.ate_between_light],
                [day.omission_pellet, day.omission_pellet_light, day.omission_pellet_between_light]
            ]:
                if cue == no_light_trial:
                    if light_on:
                        adapted_cues[i] = light_on_trial
                    else:
                        adapted_cues[i] = between_light_trial
                    break
        return adapted_cues

    def change_to_regular_opto_trials(self, day, trials):
        adapted_cues = []
        for trial in trials:
            # water trials
            if trial in [day.drank_light, day.drank_between_light]:
                adapted_cues.append(day.drank)
            elif trial in [day.not_drank_light, day.not_drank_between_light]:
                adapted_cues.append(day.not_drank)
            elif trial in [day.omission_fluid_light, day.omission_fluid_between_light]:
                adapted_cues.append(day.omission_water)

            # food trials
            elif trial in [day.ate_light, day.ate_between_light]:
                adapted_cues.append(day.ate)
            elif trial in [day.not_ate_light, day.not_ate_between_light]:
                adapted_cues.append(day.not_ate)
            elif trial in [day.omission_pellet_light, day.omission_pellet_between_light]:
                adapted_cues.append(day.omission_pellet)

            # neutral trials
            elif trial in [day.neutral_light, day.neutral_between_light]:
                adapted_cues.append(day.neutral)
            else:
                adapted_cues.append(trial)
        return np.array(adapted_cues)

    def load_increase(self, day, day_data):
        original_increases = day_data['increases']
        problem_increase = [Run(path, self.sec_hz).no_increases for path in day.runs_paths]
        manual_classification = [Run(path, self.sec_hz).ignore_roi for path in day.runs_paths]
        if sum(problem_increase):
            increases = []
            data_dicts, _ = day.load_inputs_dict_per_run()
            for i, path in enumerate(day.runs_paths):
                if Run(path, self.sec_hz).no_increases:
                    fixed_increases = []
                    y_trials = data_dicts[i]['pellet_y']
                    for trial in y_trials:
                        y_frames = np.where(np.array(trial) > 0)[0]
                        if sum(y_frames):
                            fixed_increases.append(y_frames[-1])
                        else:
                            fixed_increases.append(None)
                    increases.extend(fixed_increases)
                else:
                    increases.extend(data_dicts[i]['increases'])
            return increases
        elif sum(manual_classification):
            increases = []
            trials_classification = []
            data_dicts, _ = day.load_inputs_dict_per_run()
            for i, path in enumerate(day.runs_paths):
                trials_type = data_dicts[i]['trials_classification']
                licks = data_dicts[i]['licks_pellet']
                if Run(path, self.sec_hz).ignore_roi:
                    path_manual = join(path, 'manual_food.npy')
                    if not isfile(path_manual):
                        print('did not find manual classification')
                        sys.exit()
                    manual_classification = np.load(path_manual, allow_pickle=True)
                    food_trials_index = np.where(np.array(data_dicts[i]['cues']) == day.pellet_signal)[0]
                    for j, trial in enumerate(food_trials_index):
                        trials_type[trial] = manual_classification[j]
                    trials_classification.extend(trials_type)
                    ate_trials_index = np.where((np.array(trials_type) == day.ate) | (
                            np.array(trials_type) == day.omission_food_taste))[0]
                    new_increases = [None] * len(trials_type)
                    for trial in ate_trials_index:
                        licks_trial = [i for i in np.where(np.array(licks[trial]) == 1)[0] if i > self.sec_hz * 4]
                        if len(licks_trial):
                            new_increases[trial] = licks_trial[0]
                        else:
                            new_increases[trial] = self.sec_hz * 4
                    increases.extend(new_increases)
                else:
                    increases.extend(data_dicts[i]['increases'])
                    trials_classification.extend(trials_type)
            return increases, np.array(trials_classification)
        else:
            correct_increases = []
            for increase in original_increases:
                if increase is None:
                    correct_increases.append(None)
                elif increase < self.sec_hz * 4:
                    correct_increases.append(None)
                else:
                    correct_increases.append(increase)
            return correct_increases

    def load_brain_img(self, mouse_path, color):
        if isfile(join(mouse_path, 'brain_img', f'{color}.png')):
            image = cv2.imread(join(mouse_path, 'brain_img', f'{color}.png'), cv2.IMREAD_UNCHANGED)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            brain_img_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            flip_image = cv2.flip(brain_img_gray, 0)
            return brain_img_gray
        else:
            return []

    def load_brain_rois(self, day, identical_size=True, radius=6):
        rois, iscell, ops = day.load_suite2p_output(['stat.npy', 'iscell_final.npy', 'ops.npy'])
        relevant_cells = [i for i in range(len(rois)) if iscell[i][0] == 1.0]
        ops_dict = ops[()]
        xpix = ops_dict['Lx']
        ypix = ops_dict['Ly']
        cells_coordinates = []

        for v in relevant_cells:
            new_matrix = np.zeros([xpix, ypix])
            roi_x = rois[v]['xpix']
            roi_y = rois[v]['ypix']
            if identical_size:
                center_x = int(np.mean(roi_x))
                center_y = int(np.mean(roi_y))
                y_grid, x_grid = np.ogrid[:ypix, :xpix]
                mask = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2 <= radius ** 2
                new_matrix[mask] = 1
            else:
                lam = rois[v]['lam']
                lam_thresh = max(lam) / 5
                for pix in range(len(roi_x)):
                    if lam[pix] > lam_thresh:
                        new_matrix[roi_y[pix], roi_x[pix]] = lam[pix]
            cells_coordinates.append(new_matrix)

        return cells_coordinates

    def modulation_index(self, response1, response2):
        values1, values2, mi, opposite_cells = [], [], [], []
        perc_opposite, perc_similar = None, None
        if len(response1) > 0 and len(response2) > 0:
            for cell in range(len(response1)):
                v1 = response1[cell]
                v2 = response2[cell]
                values1.append(v1)
                values2.append(v2)

                if v1 * v2 < 0:
                    if abs(v1 / v2) >= 1.5:
                        v2 = 0
                    elif abs(v2 / v1) >= 1.5:
                        v1 = 0
                    if v1 != 0 and v2 != 0:
                        opposite_cells.append(cell)
                        continue
                mi.append((v1 - v2) / (v1 + v2))

            perc_opposite = len(opposite_cells) / len(response1) * 100
            perc_similar = len([i for i in mi if abs(i) < self.thresh_mi]) / len(response1) * 100
        return [values1, values2], mi, perc_similar, perc_opposite

    def rearrange_cells_without_baseline(self, cells):
        return [[np.hstack([cell[trial][self.sec_hz * 2:], cell[trial + 1][:self.sec_hz * 2]])
                 for trial in range(len(cell) - 1)] for cell in cells]
