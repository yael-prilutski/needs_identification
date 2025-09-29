import numpy as np
import statsmodels.api as sm
from multiprocessing import Pool

from configurations import *
from mouse import Mouse
from Processor import Processor


WEEK = 'week1'
RELEVANT_DAYS = f'{WEEK}_days'
RELEVANT_MICE = [MOUSE_YP79, MOUSE_YP82, MOUSE_YP83, MOUSE_YP84, MOUSE_YP86]


class GLM(Processor):

    def __init__(self, mouse, *args, **kwargs):
        Processor.__init__(self, *args, **kwargs)
        self.mouse = mouse
        self.days = mouse.days
        self.sec_hz = mouse.sec_hz
        self.week = mouse.week

    def single_cell_analysis(self, cell_i, iti_df, time, cumulative_time):
        cell_activity = []
        for day in self.days[:4]:
            cell_activity.extend(iti_df[day.name]['iti_df'].iloc[cell_i])
        x = sm.add_constant(time)
        glm_gauss = sm.GLM(cell_activity, x, family=sm.families.Gaussian())
        result = glm_gauss.fit()
        clean_activity = cell_activity - result.fittedvalues
        cell_day1 = clean_activity[0: cumulative_time[0]]
        cell_day2 = clean_activity[cumulative_time[0]: cumulative_time[1]]
        cell_day3 = clean_activity[cumulative_time[1]: cumulative_time[2]]
        cell_day4 = clean_activity[cumulative_time[2]: cumulative_time[3]]
        return cell_day1, cell_day2, cell_day3, cell_day4

    def run(self):
        iti_dict = np.load(
            join(self.week.data_dir_path, 'axes', f'{self.mouse.name}_iti_dict.npy'), allow_pickle=True)[()]
        n_cells = len(iti_dict[self.days[0].name]['iti_df'])

        length_days = [len(iti_dict[day.name]['iti_df'].iloc[0]) for day in self.days[:4]]
        cumulative_time = np.cumsum(length_days)
        time = np.concatenate([np.arange(l) for l in length_days])
        with Pool() as pool:
            cleaned_activity = pool.starmap(
                self.single_cell_analysis,
                [(cell, iti_dict, time, cumulative_time) for cell in range(n_cells)])

        clean_df = {
            'day1': np.array([cell[0] for cell in cleaned_activity]),
            'day2': np.array([cell[1] for cell in cleaned_activity]),
            'day3': np.array([cell[2] for cell in cleaned_activity]),
            'day4': np.array([cell[3] for cell in cleaned_activity])
        }
        np.save(join(self.week.data_dir_path, 'axes', f'{self.mouse.name}_glm_cleaned_iti_df.npy'), clean_df)


def main():
    for mouse_path in RELEVANT_MICE:
        if WEEK in mouse_path.keys():
            mouse = Mouse(mouse_path, mouse_path[RELEVANT_DAYS], mouse_path[WEEK])
            process = GLM(mouse)
            process.run()
            print(f'finished processing {mouse.name}')


'__main__' == __name__ and main()
