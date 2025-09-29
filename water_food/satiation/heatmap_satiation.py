import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
from os.path import join
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from matplotlib.colors import TwoSlopeNorm
plt.rcParams['svg.fonttype'] = 'none'

from configurations import *
from mouse import Mouse

ANALYSIS_TYPE = 'reward'
MICE = [MOUSE_YP86, MOUSE_YP84]
PATH = join(RESULTS_PATH, 'water_food', 'satiation', 'satiation_heatmaps', ANALYSIS_TYPE)


class RewardsResponsesHeatmaps:

    def __init__(self, mouse, reward):
        self.mouse = mouse
        self.week = self.mouse.week
        self.reward = reward
        self.sec_hz = mouse.days[0].smooth_factor
        self.week_data = mouse.week.data_dir_path

    def visualize(self, day_data):
        min_value = day_data.min().min() * 0.2
        max_value = day_data.max().max() * 0.2
        edge_value = min(abs(min_value), abs(max_value))
        norm = TwoSlopeNorm(vmin=-edge_value, vcenter=0, vmax=edge_value)
        color = sns.diverging_palette(255, 10, sep=90, n=150, as_cmap=True)

        plt.figure(figsize=[21, 21])
        plt.suptitle(f'{self.mouse.name} responses heatmap', fontsize=24)
        # sns.heatmap(day_data, cmap=color, vmin=-edge_value, vmax=edge_value, norm=norm)
        img = plt.imshow(day_data, cmap=color, norm=norm, aspect='auto', interpolation='nearest', rasterized=True)
        plt.colorbar(img)
        plt.yticks([])

        # plt.show()
        plt.savefig(join(PATH, f'{self.mouse.name}_{self.reward}_heatmap.svg'))
        plt.close()

    def sort_cells(self, pv_df):
        mean_values = pv_df.iloc[:, :20].mean(axis=1)
        sorted_df = pv_df.assign(mean=mean_values).sort_values('mean')
        sorted_df = sorted_df.drop('mean', axis=1)
        return sorted_df

    def run(self):
        dict_responses = np.load(
            join(self.week.data_dir_path, 'dict_mean_responses_per_trial.npy'), allow_pickle=True)[()]

        responses = dict_responses[f'day1_{ANALYSIS_TYPE}']
        smoothed_responses = pd.DataFrame([uniform_filter1d(r, 2) for r in responses])
        sorted_responses = self.sort_cells(smoothed_responses)

        self.visualize(sorted_responses)
        print(f'finished data processing {self.mouse.name}')


def main():
    for reward in ['water', 'food']:
        days = f'opto_{reward}_days'
        week = f'opto_{reward}_week'
        for mouse_dict in MICE:
            if week in mouse_dict.keys():
                mouse = Mouse(mouse_dict, mouse_dict[days], mouse_dict[week])
                processor = RewardsResponsesHeatmaps(mouse, reward)
                processor.run()


'__main__' == __name__ and main()
