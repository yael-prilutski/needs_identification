import numpy as np
from os.path import join
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.ndimage import uniform_filter1d

from configurations import *
from mouse import Mouse
from decoder.decoder_analyzer import DecoderAnalyzer

MICE = ALL_MICE
WEEK = 'week2'
DAYS = f'{WEEK}_days'


class MixedDays:

    def __init__(self, mouse, dai_i, decoder_type, path):
        self.mouse = mouse
        self.days = mouse.days
        self.path = path
        self.week_data = mouse.week.data_dir_path
        self.sec_hz = mouse.sec_hz
        self.bin = self.sec_hz
        self.chunk_size = 8 * self.sec_hz
        self.day_i = dai_i
        self.decoder_type = decoder_type
        self.analyzer = DecoderAnalyzer(self.sec_hz)

    def train_data(self, iti_dict):
        thirst_day = self.analyzer.bin_data(iti_dict[self.days[1].name]['cells'], self.bin)
        hunger_day = self.analyzer.bin_data(iti_dict[self.days[3].name]['cells'], self.bin)

        vectors_dict = np.load(join(self.week_data, 'axes', 'axes_vector_neutral.npy'), allow_pickle=True)[()]
        n_trials_thirst = self.analyzer.find_satiation_period(iti_dict[self.days[1].name],
                                                              vectors_dict[self.days[1].name]['last_index_vector'])

        thirst = thirst_day.iloc[:, :int(n_trials_thirst * self.chunk_size / self.bin)]
        hunger = hunger_day.iloc[:, :int(n_trials_thirst * self.chunk_size / self.bin)]

        x_train = np.hstack([thirst, hunger]).T
        y_train = np.hstack([
            np.ones(thirst.shape[1]),
            np.zeros(hunger.shape[1])
        ])

        if self.decoder_type == 'svm':
            clf = SVC(kernel="linear", probability=True)
        else:
            clf = LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced"
            )
        clf.fit(x_train, y_train)
        return clf

    def visualize(self, y_prob, onsets, perc_correct):
        plt.figure(figsize=(15, 15))
        plt.suptitle(f'{self.mouse.name} hunger-thirst day {self.day_i + 1}'
                     f'\nWater: {perc_correct[0]}%, Food: {perc_correct[1]}%', fontsize=20)
        plt.scatter(np.arange(len(y_prob)), y_prob, s=10)
        plt.ylabel('P(thirst)')

        water, food = onsets
        for o in water:
            plt.axvline(o, color='blue', alpha=0.3, linewidth=2)
        for o in food:
            plt.axvline(o, color='red', alpha=0.3, linewidth=2)

        plt.axhline(0.5, color='k', linestyle='--')
        plt.savefig(join(self.path, f'{WEEK}_{self.mouse.name}_hunger_thirst_day{self.day_i + 1}.jpg'))
        plt.close()

    def calculate_perc_correct(self, y_prob, onsets):
        water_o, food_o = onsets
        water_frames = np.hstack([np.arange(water_o[i], food_o[i]) for i in range(len(food_o))])
        water_o.append(len(y_prob))
        food_frames = np.hstack([np.arange(food_o[i], water_o[i + 1]) for i in range(len(food_o))])
        water_correct = len([i for i in water_frames if y_prob[i] > 0.5]) / len(water_frames) * 100
        food_correct = len([i for i in food_frames if y_prob[i] < 0.5]) / len(food_frames) * 100
        return round(water_correct, 1), round(food_correct, 1)

    def run(self):
        iti_dict = np.load(join(self.week_data, 'normalized_itis.npy'), allow_pickle=True)[()]
        clf = self.train_data(iti_dict)

        test_day = self.analyzer.bin_data(iti_dict[self.days[self.day_i].name]['cells'], self.bin)
        y_prob = clf.predict_proba(test_day.T)[:, 1]

        test_day_data = iti_dict[self.days[self.day_i].name]

        blocks_onsets = self.analyzer.blocks_onsets(test_day_data, self.days[self.day_i], self.chunk_size / self.bin)
        perc_correct = self.calculate_perc_correct(y_prob, blocks_onsets)

        self.visualize(y_prob, blocks_onsets, perc_correct)


def main():
    for decoder_type in ['logistic_regression']:
        for mouse_dict in MICE[1:]:
            if DAYS in mouse_dict.keys():
                path = join(RESULTS_PATH, decoder_type, 'mixed_days')
                mouse = Mouse(mouse_dict, mouse_dict[DAYS], mouse_dict[WEEK])
                for dai_i in [0, 2]:
                    process = MixedDays(mouse, dai_i, decoder_type, path)
                    process.run()
                print(f'finished mouse {mouse.name}')


'__main__' == __name__ and main()
