import numpy as np
from os.path import join
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

from configurations import *
from mouse import Mouse
from decoder.decoder_analyzer import DecoderAnalyzer

MICE = ALL_MICE
WEEK = 'week2'
DAYS = f'week2_days'
LINEAR = True
if not LINEAR:
    PATH = join(RESULTS_PATH, 'svm', 'non_linear_colored_after_behavior')
else:
    PATH = join(RESULTS_PATH, 'svm', 'colored_after_behavior')


class SvmSatiation:

    def __init__(self, mouse, dai_i):
        self.mouse = mouse
        self.days = mouse.days
        self.week_data = mouse.week.data_dir_path
        self.sec_hz = mouse.sec_hz
        self.smooth_factor = self.sec_hz * 3
        self.chunk_size = 3 * self.sec_hz
        self.day_i = dai_i
        self.analyzer = DecoderAnalyzer(self.sec_hz)

    def train_data(self, iti_dict):
        thirst_day = iti_dict[self.days[1].name]['iti_df']
        hunger_day = iti_dict[self.days[3].name]['iti_df']

        chunk_thirst = thirst_day.shape[1] // 4
        chunk_hunger = hunger_day.shape[1] // 4

        thirst = uniform_filter1d(thirst_day.iloc[:, :chunk_thirst], size=self.smooth_factor, axis=1)
        hunger = uniform_filter1d(hunger_day.iloc[:, :chunk_hunger], size=self.smooth_factor, axis=1)

        x_train = np.hstack([thirst, hunger]).T
        y_train = np.hstack([
            np.ones(thirst.shape[1]),
            np.zeros(hunger.shape[1])
        ])

        if LINEAR:
            clf = SVC(kernel="linear", probability=True)
        else:
            clf = SVC(kernel="rbf", probability=True)
        clf.fit(x_train, y_train)
        return clf

    def visualize(self, y_prob, onsets):
        water, food, color = onsets
        plt.figure(figsize=(15, 15))
        plt.suptitle(f'{self.mouse.name} hunger-thirst day {self.day_i}', fontsize=20)
        plt.scatter(np.arange(len(y_prob)), y_prob, color=color, s=10)
        plt.ylabel('P(thirst)')
        for o in water:
            plt.axvline(o, color='blue', alpha=0.3)
        for o in food:
            plt.axvline(o, color='red', alpha=0.3)
        plt.axhline(0.5, color='k', linestyle='--')
        plt.legend()
        plt.savefig(join(PATH, f'{self.mouse.name}_hunger_thirst_day{self.day_i}.jpg'))
        plt.close()

    def blocks_onsets(self, day_data, day):
        relevant_trials = day_data['relevant_trials']
        trials_type = day_data['trials_classification']

        food_trials = []
        water_trials = [0]

        current = 'water'
        for order_i, i in enumerate(relevant_trials[1:]):
            if trials_type[i - 1] in [
                    day.ate, day.not_ate, day.omission_pellet, day.not_ate_licked, day.omission_food_taste]:
                if current == 'water':
                    food_trials.append(order_i + 1)
                    current = 'food'

            elif trials_type[i - 1] in [day.drank, day.not_drank, day.omission_water]:
                if current != 'water':
                    water_trials.append(order_i + 1)
                    current = 'water'

        water_block = [int(i * self.chunk_size) for i in water_trials]
        food_block = [int(i * self.chunk_size) for i in food_trials]

        colors = self.analyzer.color_after_behavior(day_data, day)

        return water_block, food_block, colors

    def run(self):
        iti_dict = np.load(join(self.week_data, 'axes', f'{self.mouse.name}_iti_dict.npy'), allow_pickle=True)[()]
        clf = self.train_data(iti_dict)

        test_day = uniform_filter1d(iti_dict[self.days[self.day_i - 1].name]['iti_df'], size=self.sec_hz, axis=1)
        y_pred = clf.predict(test_day.T)
        y_prob = clf.predict_proba(test_day.T)[:, 1]
        test_day_onsets = self.blocks_onsets(iti_dict[self.days[self.day_i - 1].name], self.days[self.day_i - 1])
        self.visualize(y_prob, test_day_onsets)


def main():
    for mouse_dict in MICE:
        if DAYS in mouse_dict.keys():
            mouse = Mouse(mouse_dict, mouse_dict[DAYS], mouse_dict[WEEK])
            for dai_i in [1, 3]:
                process = SvmSatiation(mouse, dai_i)
                process.run()
            print(f'finished mouse {mouse.name}')


'__main__' == __name__ and main()
