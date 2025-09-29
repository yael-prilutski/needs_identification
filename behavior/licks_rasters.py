import numpy as np
import pandas as pd
import seaborn as sns
from os.path import join
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams['svg.fonttype'] = 'none'

from configurations import *
from mouse import Mouse
from Processor import Processor
from Visualizer import Visualizer

RELEVANT_MICE = [MOUSE_YP82]
RELEVANT_DAYS = 'week1_days'
WEEK_TYPE = 'week1'


class LicksRaster(Processor):

    def __init__(self, mouse, *args, **kwargs):
        Processor.__init__(self, *args, **kwargs)
        self.mouse = mouse
        self.sec_hz = mouse.smooth_factor
        self.visualizer = Visualizer(sec_hz=self.sec_hz)

    def plot_raster(self, day, title):
        titles = ['water', 'food', 'neutral']
        color_map = {
            1: '#1B4F72',  # water
            2: '#f70707',  # food
            4: '#17cf48'  # reward
        }

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 20))
        plt.suptitle(title, fontsize=24)

        for i in range(3):
            df = pd.DataFrame(day[i])
            ax = axes[i]
            ax.set_title(titles[i], fontsize=20)

            data = df.replace(0, np.nan).values
            rows, cols = data.shape

            for value, color in color_map.items():
                y, x = np.where(data == value)
                ax.scatter(x, y, color=color, s=10, marker='s')

            ax.set_xlabel('Time (index)', fontsize=15)
            ax.set_ylabel('Trial', fontsize=15)
            ax.set_xlim(0, cols)
            ax.set_ylim(rows, 0)
            ax.set_xticks(np.arange(0, cols, 60))
            ax.set_yticks(np.linspace(0, rows, 5))
            ax.tick_params(labelsize=12, length=4)
            ax.axvline(x=self.sec_hz * 2, color='gray', linewidth=1)
            ax.axvline(x=self.sec_hz * 4, color='gray', linewidth=1)

        plt.tight_layout()

    def visualize_licks(self, licks_scatters):
        day1, day2 = licks_scatters

        plt.figure(figsize=(27, 18))
        self.plot_raster(day1, f'thirsty {self.mouse.name}')
        plt.savefig(join(RESULTS_PATH, 'behavior', 'licks_rasters', f'thirsty_{self.mouse.name}.svg'))
        plt.close()

        plt.figure(figsize=(27, 18))
        self.plot_raster(day2, f'hungry {self.mouse.name}')
        plt.savefig(join(RESULTS_PATH, 'behavior', 'licks_rasters', f'hungry_{self.mouse.name}.svg'))
        plt.close()

    def individual_licks(self, trial):
        new_trial = np.zeros(len(trial), dtype=int)
        licks = np.where(np.array(trial) == 1)[0]
        first_licks = np.array([i for i in licks if i - 1 not in licks])
        all_indexes = [i for i in list(set(np.hstack([first_licks, first_licks + 1,  first_licks + 2])))
                       if i < len(new_trial)]
        # all_indexes = [i for i in first_licks if i < len(new_trial)]
        new_trial[all_indexes] = 1
        return new_trial

    def process_licks(self, day, day_i):
        dict_onsets = np.load(join(self.mouse.week.data_dir_path, 'rewards_indexes_onsets.npy'), allow_pickle=True)[()]
        trials_type = [day.fluid_signal, day.pellet_signal, day.neutral_signal]

        data_dict = day.load_data_dict()
        cues = data_dict['cues']
        classification = data_dict['trials_classification']
        water_licks = data_dict['water_licks']
        food_licks = data_dict['pellet_licks']

        licks_summary = []
        for trial in trials_type:
            trials_index = [i for i in np.where(cues == trial)[0] if classification[i] != day.problem]
            water_trials = [self.individual_licks(water_licks[i][:self.sec_hz * 16]) for i in trials_index]
            food_trials = [self.individual_licks(food_licks[i][:self.sec_hz * 16]) for i in trials_index]
            summary = [(water_trials[i] * 1) + (food_trials[i] * 2) for i in range(len(water_trials))]
            if trial == day.fluid_signal:
                reward_index = dict_onsets[f'day{day_i + 1}']['index_drank']
                reward_onsets = dict_onsets[f'day{day_i + 1}']['onsets_drank']
            elif trial == day.pellet_signal:
                reward_index = dict_onsets[f'day{day_i + 1}']['index_ate_full']
                reward_onsets = dict_onsets[f'day{day_i + 1}']['onsets_ate_full']
            else:
                licks_summary.append(summary)
                continue

            for i, v in enumerate(trials_index):
                if v in reward_index:
                    reward_location = reward_onsets[reward_index.index(v)]
                    summary[i][reward_location:reward_location + 4] = 4
                    summary[i][reward_location] = 4

            licks_summary.append(summary)
        return licks_summary

    def run(self):
        days = self.mouse.days
        licks_scatters = [self.process_licks(days[day], day) for day in [0, 2]]
        self.visualize_licks(licks_scatters)


def main():
    for mouse_path in RELEVANT_MICE:
        if RELEVANT_DAYS in mouse_path.keys():
            mouse = Mouse(mouse_path, mouse_path[RELEVANT_DAYS], mouse_path[WEEK_TYPE])
            process = LicksRaster(mouse)
            process.run()
            print(f'finished {mouse.name}')


'__main__' == __name__ and main()
