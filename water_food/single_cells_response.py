import numpy as np
import pandas as pd
from os import makedirs
from os.path import isdir, join
import matplotlib.pyplot as plt
from multiprocessing import Pool

from configurations import *
from mouse import Mouse
from main_analyzer import MainAnalyzer

MOUSE = MOUSE_YP79
CELL = 126
DAYS = 'week1_days'
WEEK = 'week1'


class SingleCellsResponse:

    def __init__(self, mouse):
        self.mouse = mouse
        self.days = mouse.days
        self.sec_hz = mouse.smooth_factor
        self.analyzer = MainAnalyzer(self.sec_hz)
        self.n_trials = 20
        self.cell = CELL
        self.save_path = join(RESULTS_PATH, 'water_food', 'single_cells')
        # self.save_path = join(self.mouse.week.path, 'figures', 'single_cells')
        # isdir(self.save_path) or makedirs(self.save_path)

    def cell_mean_response(self, index, cell):
        trials = pd.DataFrame([cell[i] for i in index])
        bl_sub = trials.sub(trials.iloc[:, :self.sec_hz * 2].mean(axis=1), axis=0)
        return bl_sub.mean(axis=0)[:-2 * self.sec_hz], bl_sub.std(axis=0)[:-2 * self.sec_hz] / np.sqrt(len(index))

    def visualize(self, water1, water2, water3, food1, food3, food4, cell):
        all_plots = [water1, water2, water3, food1, food3, food4]
        water_colors = ['blue', 'darkblue', 'lightblue']
        labels_water = ['day1', 'day2', 'day3']
        ate_colors = ['lightcoral', 'red', 'darkred']
        labels_ate = ['day1', 'day3', 'day4']

        max_value = max([max(p[0]) for p in all_plots]) + 0.2
        min_value = min([min(p[0]) for p in all_plots]) - 0.2

        plt.figure(figsize=(21, 18))
        plt.suptitle(f'{self.mouse.name} Cell {cell}', fontsize=30)

        plt.subplot(2, 1, 1)
        plt.title('Water Responses', fontsize=25)
        for i, plot in enumerate([water1, water2, water3]):
            plt.plot(plot[0], label=labels_water[i], color=water_colors[i])
            plt.fill_between(
                np.arange(len(plot[0])), plot[0] - plot[1], plot[0] + plot[1], color=water_colors[i], alpha=0.3)
            plt.legend(fontsize=20)
            plt.axvline(self.sec_hz * 2, color='black', linestyle='--')
            plt.axvline(self.sec_hz * 4, color='black', linestyle='--')
            plt.ylim(min_value, max_value)

        plt.subplot(2, 1, 2)
        plt.title('Food Responses', fontsize=25)
        for i, plot in enumerate([food1, food3, food4]):
            plt.plot(plot[0], label=labels_ate[i], color=ate_colors[i])
            plt.fill_between(
                np.arange(len(plot[0])), plot[0] - plot[1], plot[0] + plot[1], color=ate_colors[i], alpha=0.3)
            plt.legend(fontsize=20)
            plt.axvline(self.sec_hz * 2, color='black', linestyle='--')
            plt.axvline(self.sec_hz * 4, color='black', linestyle='--')
            plt.ylim(min_value, max_value)

        plt.savefig(join(self.save_path, f'{self.mouse.name}_cell_{cell}.svg'))
        plt.close()

    def day_response(self, day):
        day_data = day.load_data_dict()
        trials_type = day_data['trials_classification']
        # cells = self.analyzer.min_max_normalization(day_data['cells'][:300])
        # cells_drank, cells_ate = [], []
        # with Pool() as pool:
        #     if len(index_drank) > 10:
        #         cells_drank = pool.starmap(self.cell_mean_response, [(index_drank, cell) for cell in cells])
        #     if len(index_ate) > 10:
        #         cells_ate = pool.starmap(self.cell_mean_response, [(index_ate, cell) for cell in cells])

        cells = self.analyzer.min_max_single_cell(day_data['cells'][self.cell])
        index_drank = np.where(trials_type == day.drank)[0][:self.n_trials]
        index_ate = np.where(trials_type == day.ate)[0][:self.n_trials]
        cells_drank = self.cell_mean_response(index_drank, cells)
        cells_ate = self.cell_mean_response(index_ate, cells)

        return cells_drank, cells_ate

    def run_single_cell(self):
        water1, food1 = self.day_response(self.days[0])
        water2, _ = self.day_response(self.days[1])
        water3, food3 = self.day_response(self.days[2])
        _, food4 = self.day_response(self.days[3])

        self.visualize(water1, water2, water3, food1, food3, food4, self.cell)

    def run(self):
        self.run_single_cell()
        # water1, food1 = self.day_response(self.days[0])
        # water2, _ = self.day_response(self.days[1])
        # water3, food3 = self.day_response(self.days[2])
        # _, food4 = self.day_response(self.days[3])
        #
        # with Pool() as pool:
        #     pool.starmap(self.visualize,
        #                  [(water1[i], water2[i], water3[i], food1[i], food3[i], food4[i], i) for i in range(len(water1))])


def main():
    mouse = Mouse(MOUSE, MOUSE[DAYS], MOUSE[WEEK])
    process = SingleCellsResponse(mouse)
    process.run()
    print('Done')


if '__main__' == __name__:
    main()