import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from configurations import *
from mouse import Mouse
from decoder.decoder_analyzer import DecoderAnalyzer

MICE = ALL_MICE


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

    def train_data(self, iti_dict):
        day1a = iti_dict[self.days[0].name]
        day1b = iti_dict[self.days[1].name]

        binned_daya = self.decoder_analyzer.bin_data(day1a['cells'], self.sec_hz)
        binned_dayb = self.decoder_analyzer.bin_data(day1b['cells'], self.sec_hz)
        binned_full_day = pd.concat([binned_daya, binned_dayb], axis=1)

        last_ate = np.where((day1b['trials_type'] == self.days[0].ate) |
                            (day1b['trials_type'] == self.days[0].omission_food_taste))[0][-1]
        n_trials = self.decoder_analyzer.find_satiation_period(day1b, last_ate)

        chunk = int(n_trials * self.iti / self.bin)
        need = binned_full_day.iloc[:, :chunk]
        satiation = binned_full_day.iloc[:, -chunk:]

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
        elif self.decoder == 'random_forest':
            clf = RandomForestClassifier(
                n_estimators=200,  # number of trees
                max_depth=None,  # let trees grow deep
                random_state=42)
        elif self.decoder == 'neural_network':
            clf = MLPClassifier(
                hidden_layer_sizes=(100, 50),  # two hidden layers
                activation="relu",
                solver="adam",
                max_iter=500,
                random_state=42
            )
        else:
            clf = LinearDiscriminantAnalysis()
        clf.fit(x_train, y_train)
        return clf, binned_full_day

    def visualize(self, y_prob, train_day_prob):
        plt.figure(figsize=(20, 20))
        plt.suptitle(f'{self.mouse.name} full food satiation')
        plt.subplot(2, 1, 1)
        plt.title('Training day')
        plt.plot(train_day_prob, color='black')
        plt.axhline(0.5, color='k', linestyle='--')
        plt.ylabel('Need probability')

        plt.subplot(2, 1, 2)
        plt.title('Test day')
        plt.plot(y_prob, color='darkred')
        plt.axhline(0.5, color='k', linestyle='--')
        plt.ylabel('Need probability')

        plt.savefig(join(self.path, f'{self.mouse.name}_full_food_satiation.jpg'))
        plt.close()

    def run(self):
        iti_dict = np.load(join(self.week_data, 'normalized_itis.npy'), allow_pickle=True)[()]
        clf, cells_train = self.train_data(iti_dict)
        day2a = self.decoder_analyzer.bin_data(iti_dict[self.days[2].name]['cells'], self.bin)
        day2b = self.decoder_analyzer.bin_data(iti_dict[self.days[3].name]['cells'], self.bin)
        full_day2 = pd.concat([day2a, day2b], axis=1)

        y_prob = clf.predict_proba(full_day2.T)[:, 1]
        train_day_prob = clf.predict_proba(cells_train.T)[:, 1]
        self.visualize(y_prob, train_day_prob)
        return np.mean(y_prob[:int(20 * self.iti / self.bin)]), np.mean(y_prob[-int(20 * self.iti / self.bin):])


def visualize(all_mice, decoder, path):
    start_test = [m[0] for m in all_mice]
    end_test = [m[1] for m in all_mice]

    plt.figure(figsize=(10, 10))
    plt.title(f'Satiation decoder full food')
    plt.bar(['start', 'end'], [np.mean(start_test), np.mean(end_test)], color='darkred', alpha=0.6)
    for mouse in all_mice:
        plt.plot(['start', 'end'], [mouse[0], mouse[1]], color='gray', alpha=0.4)
    plt.ylim(0, 1)

    plt.savefig(join(path, f'summary_satiation_{decoder}_full_food.jpg'))
    plt.close()


def main():
    # for decoder_type in ['svm', 'logistic_regression', 'gaussian', 'lda']:
    for decoder_type in ['random_forest', 'neural_network']:
        week = f'opto_food_week'
        days = f'opto_food_days'
        path = join(RESULTS_PATH, decoder_type, 'food_satiation')
        all_mice = []
        for mouse_dict in MICE:
            if days in mouse_dict.keys():
                mouse = Mouse(mouse_dict, mouse_dict[days], mouse_dict[week])
                process = DecoderSatiation(mouse, decoder_type, path)
                all_mice.append(process.run())
                print(f'finished mouse {mouse.name}')
        visualize(all_mice, decoder_type, path)


'__main__' == __name__ and main()
