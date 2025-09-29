import numpy as np
from os import mkdir
from os.path import join, isdir
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from decoder.decoder_analyzer import DecoderAnalyzer
from scipy.ndimage import uniform_filter1d

from configurations import *
from mouse import Mouse

MICE = ALL_MICE
WEEK = 'week2'
DAYS = f'{WEEK}_days'
SMOOTH = True
SEPARATE_DAYS = True


class DecoderSatiation:

    def __init__(self, mouse, decoder, path):
        self.mouse = mouse
        self.decoder = decoder
        self.path = path
        self.days = mouse.days
        self.sec_hz = mouse.sec_hz
        self.week_data = mouse.week.data_dir_path
        self.iti = self.sec_hz * 8
        self.bin = self.sec_hz
        self.decoder_analyzer = DecoderAnalyzer(self.sec_hz)

    def train_data_water(self, day, day_name):
        vectors_dict = np.load(join(self.week_data, 'axes', 'axes_vector_neutral.npy'), allow_pickle=True)[()]
        last_trial = vectors_dict[day_name]['last_index_vector']

        binned_day = self.decoder_analyzer.bin_data(day['cells'], self.sec_hz)
        n_trials = self.decoder_analyzer.find_satiation_period(day, last_trial)
        chunk = int(n_trials * self.iti / self.bin)
        need = binned_day.iloc[:, :chunk]
        satiation = binned_day.iloc[:, -chunk:]

        x_train = np.hstack([need, satiation]).T
        y_train = np.hstack([
            np.ones(need.shape[1]),
            np.zeros(satiation.shape[1])
        ])

        if self.decoder == 'svm':
            clf = SVC(kernel="linear", probability=True)
        elif self.decoder == 'logistic_regression':
            clf = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000, class_weight="balanced")
        elif self.decoder == 'gaussian':
            clf = GaussianNB()
        else:
            clf = LinearDiscriminantAnalysis()
        clf.fit(x_train, y_train)
        return clf

    def train_data_food(self, iti_dict):
        day1a = iti_dict[self.days[3].name]
        day1b = iti_dict[self.days[4].name]

        binned_daya = self.decoder_analyzer.bin_data(day1a['cells'], self.sec_hz)
        binned_dayb = self.decoder_analyzer.bin_data(day1b['cells'], self.sec_hz)

        last_ate = np.where((day1b['trials_type'] == self.days[0].ate) |
                            (day1b['trials_type'] == self.days[0].omission_food_taste))[0][-1]
        n_trials = self.decoder_analyzer.find_satiation_period(day1b, last_ate)

        chunk = int(n_trials * self.iti / self.bin)
        need = binned_daya.iloc[:, :chunk]
        satiation = binned_dayb.iloc[:, -chunk:]

        x_train = np.hstack([need, satiation]).T
        y_train = np.hstack([
            np.ones(need.shape[1]),
            np.zeros(satiation.shape[1])
        ])

        if self.decoder == 'svm':
            clf = SVC(kernel="linear", probability=True)
        elif self.decoder == 'logistic_regression':
            clf = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000, class_weight="balanced")
        elif self.decoder == 'gaussian':
            clf = GaussianNB()
        else:
            clf = LinearDiscriminantAnalysis()
        clf.fit(x_train, y_train)
        return clf

    def visualize(self, days, blocks_onsets):
        plt.figure(figsize=(30, 20))
        plt.suptitle(f'{self.mouse.name} full food week satiation')

        for i in range(len(days)):
            water, food = days[i]
            plt.subplot(2, 2, i + 1)
            plt.title(f'Day {i + 1}')
            plt.plot(water, color='darkblue', label='Water')
            plt.plot(food, color='darkorange', label='Food')
            plt.axhline(0.5, color='k', linestyle='--')
            plt.legend()
            plt.ylabel('Need probability')
            if WEEK == 'week2':
                for onset in blocks_onsets[i][0]:
                    plt.axvline(onset, color='blue', linestyle='--', alpha=0.8, linewidth=2)
                for onset in blocks_onsets[i][1]:
                    plt.axvline(onset, color='red', linestyle='--', alpha=0.8, linewidth=2)

        plt.savefig(join(self.path, f'{self.mouse.name}_full_food_week_satiation.jpg'))
        plt.close()

    def visualize_separate(self, days, blocks_onsets):
        colors = ['darkblue', 'darkred']

        for i_reward, reward in enumerate(['water', 'food']):
            plt.figure(figsize=(30, 20))
            plt.suptitle(f'{self.mouse.name} {reward} week satiation')
            for i in range(len(days)):
                plot = days[i][i_reward]
                plt.subplot(2, 2, i + 1)
                plt.title(f'Day {i + 1}')
                plt.plot(plot, color=colors[i_reward])
                plt.axhline(0.5, color='k', linestyle='--')
                plt.ylabel(f'{reward} probability')
                if WEEK == 'week2':
                    for onset in blocks_onsets[i][0]:
                        plt.axvline(onset, color='blue', linestyle='--', alpha=0.8, linewidth=2)
                    for onset in blocks_onsets[i][1]:
                        plt.axvline(onset, color='red', linestyle='--', alpha=0.8, linewidth=2)

            plt.savefig(join(self.path, f'{self.mouse.name}_{reward}_satiation.jpg'))
            plt.close()

    def run(self):
        iti_dict = np.load(join(self.week_data, 'normalized_itis.npy'), allow_pickle=True)[()]
        clf_water = self.train_data_water(iti_dict[self.days[1].name], self.days[1].name)
        clf_food = self.train_data_food(iti_dict)

        decoded_days = []
        blocks_onsets = []
        for day in self.days[:4]:
            binned_day = self.decoder_analyzer.bin_data(iti_dict[day.name]['cells'], self.bin)
            water_prob = clf_water.predict_proba(binned_day.T)[:, 1]
            food_prob = clf_food.predict_proba(binned_day.T)[:, 1]
            if SMOOTH:
                water_prob = uniform_filter1d(water_prob, size=8)
                food_prob = uniform_filter1d(food_prob, size=8)
            onsets = self.decoder_analyzer.blocks_onsets(iti_dict[day.name], day, self.iti // self.bin)
            decoded_days.append([water_prob, food_prob])
            blocks_onsets.append(onsets)

        if SEPARATE_DAYS:
            self.visualize_separate(decoded_days, blocks_onsets)
        else:
            self.visualize(decoded_days, blocks_onsets)
        return decoded_days, blocks_onsets


# def mice_visualize(mice, path):
#     mice_days = [m[0] for m in mice]
#     onsets = [m[1] for m in mice]
#     colors = ['darkblue', 'darkred']
#
#     for i_reward, reward in enumerate(['water', 'food']):
#         plt.figure(figsize=(30, 20))
#         plt.suptitle(f'Summary {reward} week satiation')
#         for i in range(len(days)):
#             plot = days[i][i_reward]
#             plt.subplot(2, 2, i + 1)
#             plt.title(f'Day {i + 1}')
#             plt.plot(plot, color=colors[i_reward])
#             plt.axhline(0.5, color='k', linestyle='--')
#             plt.ylabel(f'{reward} probability')
#             if WEEK == 'week2':
#                 for onset in blocks_onsets[i][0]:
#                     plt.axvline(onset, color='blue', linestyle='--', alpha=0.8, linewidth=2)
#                 for onset in blocks_onsets[i][1]:
#                     plt.axvline(onset, color='red', linestyle='--', alpha=0.8, linewidth=2)
#
#         plt.savefig(join(self.path, f'{self.mouse.name}_{reward}_satiation.jpg'))
#         plt.close()


def main():
    for decoder_type in ['svm']:
        all_mice = []
        path = join(RESULTS_PATH, decoder_type, 'week_satiation', 'full_food')
        if WEEK == 'week1':
            path = join(RESULTS_PATH, decoder_type, 'week_satiation', 'week1', 'full_food')
        if SEPARATE_DAYS:
            path = join(path, 'separate_days')
        if SMOOTH:
            path = join(path, 'smoothed')
        isdir(path) or mkdir(path)
        for mouse_dict in MICE:
            if DAYS in mouse_dict.keys():
                mouse = Mouse(mouse_dict, mouse_dict[DAYS], mouse_dict[WEEK])
                process = DecoderSatiation(mouse, decoder_type, path)
                all_mice.append(process.run())
                print(f'finished mouse {mouse.name}')
        # mice_visualize(all_mice, path)


'__main__' == __name__ and main()
