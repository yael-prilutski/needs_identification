import numpy as np
import pandas as pd
from os import makedirs
from os.path import isdir, join
import matplotlib.pyplot as plt
from multiprocessing import Pool
plt.rcParams['svg.fonttype'] = 'none'

from configurations import *
from mouse import Mouse
from main_analyzer import MainAnalyzer

MOUSE = MOUSE_YP82
REWARD = 'water'
DAYS = f'opto_{REWARD}_days'
WEEK = f'opto_{REWARD}_week'
PATH = r'\\isi.storwis.weizmann.ac.il\Labs\livneh\yaelpri\manuscript\figures\water_food\satiation\single_cells_satiety'
CELL = 28


class SingleCellsResponse:

    def __init__(self, mouse, reward):
        self.mouse = mouse
        self.week = mouse.week
        self.days = mouse.days
        self.sec_hz = mouse.smooth_factor
        self.analyzer = MainAnalyzer(self.sec_hz)
        self.reward = reward
        if reward == 'water':
            self.colors = ['darkblue', 'lightblue']
        else:
            self.colors = ['darkred', 'lightcoral']
        self.n_trials = 20
        # self.save_path = join(self.mouse.week.path, 'figures', 'single_cells_satiety')
        # isdir(self.save_path) or makedirs(self.save_path)

    def cell_mean_response(self, index, cell):
        trials = pd.DataFrame([cell[i] for i in index])
        bl_sub = trials.sub(trials.iloc[:, :self.sec_hz * 2].mean(axis=1), axis=0)
        start_day_mean = bl_sub.iloc[:self.n_trials].mean(axis=0)[:-2 * self.sec_hz]
        start_day_std = bl_sub.iloc[:self.n_trials].std(axis=0)[:-2 * self.sec_hz] / np.sqrt(self.n_trials)
        end_day_mean = bl_sub.iloc[-self.n_trials:].mean(axis=0)[:-2 * self.sec_hz]
        end_day_std = bl_sub.iloc[-self.n_trials:].std(axis=0)[:-2 * self.sec_hz] / np.sqrt(self.n_trials)
        return [start_day_mean, start_day_std], [end_day_mean, end_day_std]

    def visualize(self, response, cell):
        labels = ['start', 'end']

        plt.figure(figsize=(21, 18))
        plt.suptitle(f'{self.mouse.name} {self.reward} responses, Cell {cell}', fontsize=30)

        for i in range(2):
            plt.plot(response[i][0], label=labels[i], color=self.colors[i])
            plt.fill_between(
                np.arange(len(response[i][0])), response[i][0] - response[i][1], response[i][0] + response[i][1],
                color=self.colors[i], alpha=0.3)
            plt.legend(fontsize=20)
            plt.axvline(self.sec_hz * 2, color='black', linestyle='--')
            plt.axvline(self.sec_hz * 4, color='black', linestyle='--')

        plt.savefig(join(PATH, f'{self.mouse.name}_cell_{cell}.svg'))
        plt.close()

    def day_response(self, day):
        cell = self.analyzer.min_max_single_cell(day.load_data_dict()['cells'][CELL])
        if self.reward == 'water':
            index = np.load(join(self.week.data_dir_path, 'water_licks_onsets.npy'),
                            allow_pickle=True)[()][day.name]['drank_index']
        else:
            index = np.load(join(self.week.data_dir_path, 'food_consumption_onsets.npy'),
                            allow_pickle=True)[()][day.name]['ate_index']

        # with Pool() as pool:
        #     cells_responses = pool.starmap(self.cell_mean_response, [(index, cell) for cell in cells])
        # return cells_responses
        return self.cell_mean_response(index, cell)

    def run(self):
        day_response = self.day_response(self.days[0])
        self.visualize(day_response, CELL)
        # with Pool() as pool:
        #     pool.starmap(self.visualize, [(day_response[i], i) for i in range(len(day_response))])


def main():
    mouse = Mouse(MOUSE, MOUSE[DAYS], MOUSE[WEEK])
    process = SingleCellsResponse(mouse, REWARD)
    process.run()
    print('Done')


if '__main__' == __name__:
    main()