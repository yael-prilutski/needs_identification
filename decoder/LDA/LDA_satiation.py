import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from configurations import *
from mouse import Mouse

MICE = ALL_MICE
PATH = join(RESULTS_PATH, 'LDA', 'satiation')


class LdaSatiation:

    def __init__(self, mouse, reward):
        self.mouse = mouse
        self.days = mouse.days[:2]
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

        clf = LinearDiscriminantAnalysis()
        clf.fit(x_train, y_train)
        return clf

    def visualize(self, y_prob):
        plt.suptitle(f'{self.mouse.name} {self.reward} satiation')
        plt.plot(y_prob, label="Need probability", color=self.color)
        plt.axhline(0.5, color='k', linestyle='--')
        plt.legend()
        plt.savefig(join(PATH, f'{self.mouse.name}_{self.reward}_satiation.jpg'))
        plt.close()

    def run(self):
        iti_dict = np.load(join(self.week_data, 'axes', f'{self.mouse.name}_iti_dict.npy'), allow_pickle=True)[()]
        clf = self.train_data(iti_dict[self.days[0].name]['iti_df'])
        day2 = uniform_filter1d(iti_dict[self.days[1].name]['iti_df'], size=self.smooth_factor, axis=1)
        y_prob = clf.predict_proba(day2.T)[:, 1]
        self.visualize(y_prob)


def main():
    for reward in ['water', 'food']:
        week = f'opto_{reward}_week'
        days = f'opto_{reward}_days'
        for mouse_dict in MICE:
            if days in mouse_dict.keys():
                mouse = Mouse(mouse_dict, mouse_dict[days], mouse_dict[week])
                process = LdaSatiation(mouse, reward)
                process.run()
                print(f'finished mouse {mouse.name}')


'__main__' == __name__ and main()
