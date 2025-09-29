import numpy as np
import pandas as pd
from os import mkdir
from os.path import join, isdir
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from configurations import *
from mouse import Mouse
from decoder.decoder_analyzer import DecoderAnalyzer

MICE = ALL_MICE
WEEK = 'week2'
DAYS = f'{WEEK}_days'
PATH = join(RESULTS_PATH, 'PCA')
if WEEK == 'week1':
    PATH = join(PATH, 'week1')
isdir(PATH) or mkdir(PATH)


class DecoderSatiation:

    def __init__(self, mouse):
        self.mouse = mouse
        self.days = mouse.days
        self.sec_hz = mouse.sec_hz
        self.week_data = mouse.week.data_dir_path
        self.iti = self.sec_hz * 8
        self.bin = self.sec_hz
        self.decoder_analyzer = DecoderAnalyzer(self.sec_hz)

    def analyze_day(self, iti_dict, day_i):
        relevant_days = [1, day_i, 3]
        binned_data = [self.decoder_analyzer.bin_data(iti_dict[self.days[i].name]['cells'], self.bin)
                       for i in relevant_days]
        concatenated = pd.concat(binned_data, axis=1, ignore_index=True).T
        length = [len(d.iloc[0]) for d in binned_data]
        onsets = self.decoder_analyzer.blocks_onsets(
            iti_dict[self.days[day_i].name], self.days[day_i], self.iti // self.bin)

        pca = PCA(n_components=10)
        projections = pca.fit_transform(concatenated).T
        explained = pca.explained_variance_ratio_

        return projections, explained, length, onsets

    def visualize(self, day_data, day_i):
        components, explained, length, onsets = day_data
        plt.figure(figsize=(30, 20))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.plot(components[i])
            plt.title(f'component: {i}, {round(explained[i], 3)}')
            for o in np.cumsum(length)[:-1]:
                plt.axvline(o, color='red', linestyle='--')
        plt.savefig(join(PATH, f'{self.mouse.name}_pca_day{day_i}.jpg'))
        plt.close()

        start = length[0]
        end = start + length[1]
        plt.figure(figsize=(20, 15))
        plt.suptitle(f'{self.mouse.name} PCA mixed day {day_i}')
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.title(f'component: {i}, {round(explained[i], 3)}')
            plt.plot(components[i][start:end])
            for o in onsets[0]:
                plt.axvline(o, color='blue', linestyle='--', alpha=0.5, linewidth=2)
            for o in onsets[1]:
                plt.axvline(o, color='red', linestyle='--', alpha=0.5, linewidth=2)
        plt.savefig(join(PATH, 'mix_only', f'{self.mouse.name}_pca_day{day_i}_mixed.jpg'))

    def run(self):
        iti_dict = np.load(join(self.week_data, 'normalized_itis.npy'), allow_pickle=True)[()]
        day1 = self.analyze_day(iti_dict, 0)
        day3 = self.analyze_day(iti_dict, 2)
        self.visualize(day1, 1)
        self.visualize(day3, 3)


def main():
    for mouse_dict in MICE:
        if DAYS in mouse_dict.keys():
            mouse = Mouse(mouse_dict, mouse_dict[DAYS], mouse_dict[WEEK])
            process = DecoderSatiation(mouse)
            process.run()
            print(f'finished mouse {mouse.name}')


'__main__' == __name__ and main()
