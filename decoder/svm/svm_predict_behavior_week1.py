import numpy as np
from os.path import join
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

from configurations import *
from mouse import Mouse

MICE = ALL_MICE
DAYS = 'week1_days'
WEEK = 'week1'
PATH = join(RESULTS_PATH, 'svm', 'colored_predict_behavior', WEEK)


class SvmSatiation:

    def __init__(self, mouse, day_i):
        self.mouse = mouse
        self.days = mouse.days
        self.week_data = mouse.week.data_dir_path
        self.sec_hz = mouse.sec_hz
        self.smooth_factor = self.sec_hz * 3
        self.chunk_size = 3 * self.sec_hz
        self.day_i = day_i

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

        clf = SVC(kernel="linear", probability=True)
        clf.fit(x_train, y_train)
        return clf

    def visualize(self, y_prob, colors):
        plt.figure(figsize=(15, 15))
        plt.suptitle(f'{self.mouse.name} hunger-thirst day {self.day_i}', fontsize=20)
        plt.scatter(np.arange(len(y_prob)), y_prob, color=colors, s=10)
        plt.ylabel('P(thirst)')
        plt.axhline(0.5, color='k', linestyle='--')
        plt.legend()
        plt.savefig(join(PATH, f'week1_{self.mouse.name}_hunger_thirst_day{self.day_i}.jpg'))
        plt.close()

    def color_trials(self, day_data, day):
        relevant_trials = day_data['relevant_trials']
        trials_type = day_data['trials_classification']
        colors = []
        for i in relevant_trials:
            if i == relevant_trials[-1]:
                colors.extend(['gray'] * self.chunk_size)
                continue
            trial = trials_type[i + 1]
            if trial == day.neutral:
                colors.extend(['gray'] * self.chunk_size)
            elif trial in [day.ate, day.omission_food_taste]:
                colors.extend(['red'] * self.chunk_size)
            elif trial in [day.not_ate, day.not_ate_licked]:
                colors.extend(['darkred'] * self.chunk_size)
            elif trial == day.omission_pellet:
                colors.extend(['brown'] * self.chunk_size)
            elif trial == day.drank:
                colors.extend(['lightblue'] * self.chunk_size)
            elif trial == day.not_drank:
                colors.extend(['darkblue'] * self.chunk_size)
            elif trial == day.omission_water:
                colors.extend(['green'] * self.chunk_size)
            else:
                colors.extend(['black'] * self.chunk_size)
        return colors

    def run(self):
        iti_dict = np.load(join(self.week_data, 'axes', f'{self.mouse.name}_iti_dict.npy'), allow_pickle=True)[()]
        clf = self.train_data(iti_dict)

        test_day = uniform_filter1d(iti_dict[self.days[self.day_i - 1].name]['iti_df'], size=self.sec_hz, axis=1)
        y_pred = clf.predict(test_day.T)
        y_prob = clf.predict_proba(test_day.T)[:, 1]
        colors = self.color_trials(iti_dict[self.days[self.day_i - 1].name], self.days[self.day_i - 1])
        self.visualize(y_prob, colors)


def main():
    for mouse_dict in MICE:
        if DAYS in mouse_dict.keys():
            mouse = Mouse(mouse_dict, mouse_dict[DAYS], mouse_dict[WEEK])
            for day in [1, 3]:
                process = SvmSatiation(mouse, day)
                process.run()
                print(f'finished mouse {mouse.name}')


'__main__' == __name__ and main()
