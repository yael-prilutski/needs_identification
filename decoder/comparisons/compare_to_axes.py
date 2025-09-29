import numpy as np
import pandas as pd
from multiprocessing import Pool
from os.path import join, isdir
import matplotlib.pyplot as plt

from configurations import *
from mouse import Mouse
from Processor import Processor
from main_analyzer import MainAnalyzer

RELEVANT_DAYS = ['opto_water_days']
RELEVANT_MICE = [MOUSE_SS38]
N_DFF = 10


class CompareNormalization(Processor):

    def __init__(self, day, *args, **kwargs):
        Processor.__init__(self, *args, **kwargs)
        self.day = day
        self.sec_hz = self.day.sec_hz
        self.chunk = self.sec_hz * 3
        self.main_analyzer = MainAnalyzer(self.sec_hz)

    def visualize(self, axes_data, sw_data, zscore_data):
        for cell in range(70):
            cell_axes = axes_data.iloc[cell]
            cell_iti = sw_data[cell]

            plt.figure(figsize=(30, 10))
            plt.title(f'Cell {cell}')
            plt.plot(cell_axes, color='darkblue', alpha=0.6, label='axes')
            plt.plot(cell_iti, color='darkred', alpha=0.6, label='sw_min_max')
            # plt.plot(zscore_data[cell], color='green', alpha=0.6, label='zscore')
            plt.legend()

            plt.savefig(join(RESULTS_PATH, 'axes_sw_comparison', f'{self.day.mouse_name}_{self.day.name}_cell{cell}.jpg'))
            plt.close()

    def run(self, iti_dict):
        axes_data = iti_dict['iti_df']
        sw_data = np.load(join(self.day.data_dir_path, f'{self.day.mouse_name}_{self.day.name}_itis_sw_dff.npy'),
                          allow_pickle=True)[()]
        relevant_keys = [k for k in sw_data.keys() if 'run' in k]
        runs_concatenated = [pd.concat([sw_data[k].iloc[cell] for k in relevant_keys],
                                       ignore_index=True) for cell in range(len(sw_data['run1']))]
        normalized_min_max = []
        normalized_zscore = []
        for cell in runs_concatenated:
            min_cell = np.percentile(cell, 1)
            max_cell = np.percentile(cell, 99)
            normalized_cell = (cell - min_cell) / (max_cell - min_cell)
            normalized_min_max.append(normalized_cell)

            mean_cell = np.mean(cell)
            std_cell = np.std(cell)
            zscored_cell = (cell - mean_cell) / std_cell
            normalized_zscore.append(zscored_cell)
        self.visualize(axes_data, normalized_min_max, normalized_zscore)


def main(mice_paths, relevant_days):
    for m in mice_paths:
        for days in relevant_days:
            if days in m.keys():
                mouse = Mouse(m, m[days], m['opto_water_week'])
                iti_dict = np.load(
                    join(mouse.week.data_dir_path, 'axes', f'{mouse.name}_iti_dict.npy'), allow_pickle=True)[()]
                process = CompareNormalization(mouse.days[0])
                process.run(iti_dict[mouse.days[0].name])


'__main__' == __name__ and main(RELEVANT_MICE, RELEVANT_DAYS)
