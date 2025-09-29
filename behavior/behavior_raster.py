import numpy as np
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams['svg.fonttype'] = 'none'

from configurations import *
from mouse import Mouse
from Processor import Processor
from Visualizer import Visualizer

RELEVANT_MICE = [MOUSE_YP79]
RELEVANT_DAYS = 'week1_days'
WEEK_TYPE = 'week1'


class LicksRaster(Processor):

    def __init__(self, mouse, *args, **kwargs):
        Processor.__init__(self, *args, **kwargs)
        self.mouse = mouse
        self.sec_hz = mouse.smooth_factor
        self.visualizer = Visualizer(sec_hz=self.sec_hz)

    def plot_raster(self, day, title):
        titles = ['Water', 'Food']
        color_map = {
            0: '#f2f2f2',
            1: '#fa0505',
            2: '#17cf48',
            3: '#13c1e8'
        }

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 20))
        plt.suptitle(title, fontsize=24)

        for i in range(2):
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
        self.plot_raster(day1, f"Thirsty {self.mouse.name}")
        plt.savefig(join(RESULTS_PATH, 'behavior', 'behavior_rasters', f'thirsty_{self.mouse.name}.svg'))
        plt.close()

        plt.figure(figsize=(27, 18))
        self.plot_raster(day2, f"Hungry {self.mouse.name}")
        plt.savefig(join(RESULTS_PATH, 'behavior', 'behavior_rasters', f'hungry_{self.mouse.name}.svg'))
        plt.close()

    def individual_licks(self, trial):
        new_trial = np.zeros(len(trial), dtype=int)
        licks = np.where(np.array(trial) == 1)[0]
        first_licks = np.array([i for i in licks if i - 1 not in licks])
        all_indexes = [i for i in list(set(np.hstack([first_licks, first_licks + 1,  first_licks + 2])))
                       if i < len(new_trial)]
        new_trial[all_indexes] = 1
        return new_trial

    def process_licks(self, day, day_i):
        dict_onsets = np.load(join(self.mouse.week.data_dir_path, 'rewards_indexes_onsets.npy'), allow_pickle=True)[()]
        trials_type = [day.fluid_signal, day.pellet_signal]

        data_dict = day.load_data_dict()
        cues = data_dict['cues']
        classification = data_dict['trials_classification']
        pellet_y = data_dict['pellet_y']

        licks_summary = []
        for trial in trials_type:
            trials_index = [i for i in np.where(cues == trial)[0] if classification[i] != day.problem]
            summary = np.zeros((len(trials_index), pd.DataFrame(pellet_y).shape[1]))
            if trial == day.fluid_signal:
                reward_index = dict_onsets[f'day{day_i + 1}']['index_drank']
                reward_onsets = dict_onsets[f'day{day_i + 1}']['onsets_drank']
                onsets_delivery = [self.sec_hz * 4 + 2] * len(trials_index)
            else:
                reward_index = dict_onsets[f'day{day_i + 1}']['index_ate_full']
                reward_onsets = dict_onsets[f'day{day_i + 1}']['onsets_ate_full']
                onsets_delivery = [np.where(np.array(pellet_y[i]) > 0)[0][0] for i in trials_index]

            for i in range(len(trials_index)):
                summary[i][onsets_delivery[i]: onsets_delivery[i] + 3] = 1

            for i, v in enumerate(trials_index):
                if v in reward_index:
                    reward_location = reward_onsets[reward_index.index(v)]
                    summary[i][reward_location:reward_location + 4] = 2

            if trial == day.pellet_signal:
                bb_down_runs = [d['bb_down'] for d in day.load_inputs_dict_per_run()[0]]
                bb_down_trials = [item for sublist in bb_down_runs for item in sublist]
                bb_down_frames = []
                for t in bb_down_trials:
                    frames = np.where(np.array(t) == 1)[0]
                    if len(frames) > 0:
                        bb_down_frames.append(frames[0])
                    else:
                        bb_down_frames.append(None)

                for i, v in enumerate(trials_index):
                    if bb_down_frames[v] is not None:
                        summary[i][bb_down_frames[v]:bb_down_frames[v] + 4] = 3

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
