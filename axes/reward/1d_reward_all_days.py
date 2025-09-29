import numpy as np
import pandas as pd
from os.path import join
from multiprocessing import Pool
import matplotlib.pyplot as plt

from configurations import *
from mouse import Mouse
from Processor import Processor
from axes.axes_analyzer import AxesAnalyzer
from main_analyzer import MainAnalyzer

WEEK = 'week1'
RELEVANT_DAYS = f'{WEEK}_days'
RELEVANT_MICE = ALL_MICE
N_TRIALS = 30
PATH = join(RESULTS_PATH, 'axes', 'reward', '1d_reward_all_days', WEEK)


class AllWeekCueAxes(Processor):

    def __init__(self, mouse, vector_name, *args, **kwargs):
        Processor.__init__(self, *args, **kwargs)
        self.mouse = mouse
        self.days = mouse.days
        self.vector_name = vector_name
        self.sec_hz = mouse.smooth_factor
        self.week = mouse.week
        self.analyzer = AxesAnalyzer(mouse.smooth_factor)
        self.main_analyzer = MainAnalyzer(mouse.smooth_factor)

    def visualize(self, trials_by_axes):
        titles_days = ['day1', 'day2', 'day3', 'day4']
        titles_trial_type = ['drank', 'not drank', 'ate', 'not ate', 'neutral']
        titles_axes = ['water', 'food']
        colors_axes = ['darkblue', 'darkred']

        current_plot = 1
        plt.figure(figsize=(30, 30))
        plt.suptitle('Cue Axes Responses', fontsize=30)
        for day in range(4):
            day_name = titles_days[day]
            data = trials_by_axes[day]
            for trial_type in range(5):
                trial_name = titles_trial_type[trial_type]
                trial_data = data[trial_type]
                plt.subplot(4, 5, current_plot)
                current_plot += 1
                plt.title(f'{day_name} {trial_name}', fontsize=20)
                if len(trial_data) > 0:
                    for axis in range(2):
                        plt.plot(trial_data[axis], label=titles_axes[axis], color=colors_axes[axis], alpha=0.7)
                    plt.axvline(x=self.sec_hz * 2, color='black', linestyle='--')
                    plt.axvline(x=self.sec_hz * 4, color='black', linestyle='--')
                    plt.legend()
                    plt.yticks(fontsize=15)

        plt.savefig(join(PATH, f'{self.mouse.name}_1d_reward_axes.jpg'))
        plt.close()

    def select_response(self, cell, index):
        mean_responses = pd.DataFrame([cell[i][:self.sec_hz * 10] for i in index]).mean(axis=0)
        return mean_responses

    def load_day_responses(self, day, vectors):
        test_trials = [day.drank, day.not_drank, [day.ate, day.omission_food_taste], [day.not_ate, day.not_ate_licked],
                       day.neutral]
        vector_types = ['water', 'food']
        day_data = day.load_data_dict()
        normalized_cells = self.main_analyzer.min_max_normalization(day_data['cells'])
        trials_type = day_data['trials_classification']

        all_trials_by_axes = []
        for trial in test_trials:
            if type(trial) is not list:
                relevant_trials = np.where(trials_type == trial)[0]
            else:
                relevant_trials = np.where((trials_type == trial[0]) | (trials_type == trial[1]))[0]
            if len(relevant_trials) > 0:
                with Pool() as pool:
                    trial_responses = pool.starmap(self.select_response, [(cell, relevant_trials[:N_TRIALS])
                                                                          for cell in normalized_cells])
                    all_axes_products = [self.analyzer.create_dot_product_iti(
                        pd.DataFrame(trial_responses), vectors[vector_type]) for vector_type in vector_types]
                    all_trials_by_axes.append(all_axes_products)
            else:
                all_trials_by_axes.append([])

        return all_trials_by_axes

    def run(self):
        vectors = np.load(join(self.week.data_dir_path, 'axes', self.vector_name), allow_pickle=True)[()]
        days_trials_axes = [self.load_day_responses(d, vectors) for d in self.days[:4]]
        self.visualize(days_trials_axes)


def main():
    for mouse_path in RELEVANT_MICE:
        if WEEK in mouse_path.keys():
            mouse = Mouse(mouse_path, mouse_path[RELEVANT_DAYS], mouse_path[WEEK])
            process = AllWeekCueAxes(mouse, vector_name='reward_baseline_axes.npy')
            process.run()
            print(f'finished processing {mouse.name}')


'__main__' == __name__ and main()
