import numpy as np
from multiprocessing import Pool

from configurations import *
from mouse import Mouse

# RELEVANT_DAYS = [['week1_days', 'week1'], ['week2_days', 'week2'], ['opto_water_days', 'opto_water_week'],
#                  ['opto_food_days', 'opto_food_week']]
RELEVANT_DAYS = [['opto_water_days', 'opto_water_week']]
RELEVANT_MICE = ALL_MICE


class NormalizeITIs:

    def __init__(self, mouse, days):
        self.mouse = mouse
        self.days = days
        self.sec_hz = mouse.sec_hz

    @staticmethod
    def normalize_run_min_max(cell):
        min_cell = np.percentile(cell, 1)
        max_cell = np.percentile(cell, 99)
        normalized_cell = (cell - min_cell) / (max_cell - min_cell)
        return normalized_cell

    @staticmethod
    def normalize_run_zscore(cell):
        mean_cell = np.mean(cell)
        std_cell = np.std(cell)
        normalized_cell = (cell - mean_cell) / std_cell
        return normalized_cell

    def run(self):
        dict_days = {}
        with Pool() as pool:
            for day in self.days:
                # full_cells = day.load_data_dict()
                iti_dict = np.load(join(day.data_dir_path, f'{day.mouse_name}_{day.name}_itis_sw_dff.npy'),
                                   allow_pickle=True)[()]
                relevant_keys = [k for k in iti_dict.keys() if 'run' in k]
                combined_day = [np.concatenate([iti_dict[run].iloc[cell] for run in relevant_keys]) for cell in
                                range(len(iti_dict[relevant_keys[0]]))]
                normalized_day = pool.map(self.normalize_run_min_max, combined_day)
                dict_days[day.name] = {
                    'cells': normalized_day,
                    'relevant_trials': iti_dict['relevant_trials'],
                    'trials_type': iti_dict['trials_type']
                }
        np.save(join(self.mouse.week.data_dir_path, 'normalized_itis.npy'), dict_days)


def main(mice_paths, days_to_run):
    for m in mice_paths:
        for days, week in days_to_run:
            if days in m.keys():
                mouse = Mouse(m, m[days], m[week])
                if 'opto' in days:
                    relevant_days = mouse.days
                else:
                    relevant_days = mouse.days[:5]
                process = NormalizeITIs(mouse, relevant_days)
                process.run()
                print(f'finished mouse {mouse.name} {week}')


if '__main__' == __name__:
    main(RELEVANT_MICE, RELEVANT_DAYS)
