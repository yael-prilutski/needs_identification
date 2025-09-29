import numpy as np
from os.path import join
import pandas as pd
from scipy.ndimage import uniform_filter1d

from sklearn.svm import SVC
import matplotlib.pyplot as plt

from configurations import *
from mouse import Mouse
from decoder.decoder_analyzer import DecoderAnalyzer

MICE = ALL_MICE
PATH = join(RESULTS_PATH, 'svm', 'satiation')


class SvmSatiation:

    def __init__(self, mouse, reward):
        self.mouse = mouse
        self.days = mouse.days[:2]
        self.sec_hz = mouse.sec_hz
        self.week_data = mouse.week.data_dir_path
        self.smooth_factor = mouse.sec_hz * 10
        self.reward = reward
        if reward == 'water':
            self.color = 'darkblue'
        else:
            self.color = 'darkred'

    def train_data(self, day1):
        chunk = day1.shape[1] // 5
        need = uniform_filter1d(day1.iloc[:, :chunk], size=self.smooth_factor, axis=1)
        satiation = uniform_filter1d(day1.iloc[:, -chunk:], size=self.smooth_factor, axis=1)

        x_train = np.hstack([need, satiation]).T
        y_train = np.hstack([
            np.ones(need.shape[1]),
            np.zeros(satiation.shape[1])
        ])

        clf = SVC(kernel="linear", probability=True)
        clf.fit(x_train, y_train)
        return clf

    def visualize(self, y_prob, train_day_prob):
        plt.figure(figsize=(20, 20))
        plt.suptitle(f'{self.mouse.name} {self.reward} satiation')
        plt.subplot(2, 1, 1)
        plt.title('Training day')
        plt.plot(train_day_prob, color='black')
        plt.axhline(0.5, color='k', linestyle='--')
        plt.ylabel('Need probability')

        plt.subplot(2, 1, 2)
        plt.title('Test day')
        plt.plot(y_prob, color=self.color)
        plt.axhline(0.5, color='k', linestyle='--')
        plt.ylabel('Need probability')

        plt.savefig(join(PATH, f'{self.mouse.name}_{self.reward}_satiation.jpg'))
        plt.close()

    def run(self):
        iti_dict = np.load(join(self.week_data, 'normalized_itis.npy'), allow_pickle=True)[()]
        clf = self.train_data(pd.DataFrame(iti_dict[self.days[0].name]['cells']))

        cells_train = uniform_filter1d(pd.DataFrame(iti_dict[self.days[0].name]['cells']), size=self.sec_hz, axis=1)
        day2 = uniform_filter1d(pd.DataFrame(iti_dict[self.days[1].name]['cells']), size=self.smooth_factor, axis=1)
        y_pred = clf.predict(day2.T)
        y_prob = clf.predict_proba(day2.T)[:, 1]
        train_day_prob = clf.predict_proba(cells_train.T)[:, 1]
        self.visualize(y_prob, train_day_prob)


def main():
    for reward in ['water', 'food']:
        week = f'opto_{reward}_week'
        days = f'opto_{reward}_days'
        for mouse_dict in MICE:
            if days in mouse_dict.keys():
                mouse = Mouse(mouse_dict, mouse_dict[days], mouse_dict[week])
                process = SvmSatiation(mouse, reward)
                process.run()
                print(f'finished mouse {mouse.name}')


'__main__' == __name__ and main()
