import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from decoder.decoder_analyzer import DecoderAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from configurations import *
from mouse import Mouse

MICE = ALL_MICE


class DecoderSatiation:

    def __init__(self, mouse, reward, decoder, path):
        self.mouse = mouse
        self.decoder = decoder
        self.path = path
        self.days = [d for d in mouse.days if d.name[-1] != 'b']
        self.sec_hz = mouse.sec_hz
        self.week_data = mouse.week.data_dir_path
        self.iti = self.sec_hz * 8
        self.bin = self.sec_hz
        self.reward = reward
        self.decoder_analyzer = DecoderAnalyzer(self.sec_hz)

    def train_data(self, day1):
        vectors_dict = np.load(join(self.week_data, 'axes', 'axes_vector_neutral.npy'), allow_pickle=True)[()]
        last_trial = vectors_dict[self.days[0].name]['last_index_vector']

        binned_day = self.decoder_analyzer.bin_data(day1['cells'], self.sec_hz)
        n_trials = self.decoder_analyzer.find_satiation_period(day1, last_trial)
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
                random_state=42)
        else:
            clf = LinearDiscriminantAnalysis()
        clf.fit(x_train, y_train)
        return clf

    def run(self):
        iti_dict = np.load(join(self.week_data, 'normalized_itis.npy'), allow_pickle=True)[()]
        clf = self.train_data(iti_dict[self.days[0].name])
        day2 = self.decoder_analyzer.bin_data(iti_dict[self.days[1].name]['cells'], self.bin)

        y_prob = clf.predict_proba(day2.T)[:, 1]
        return np.mean(y_prob[:int(20 * self.iti / self.bin)]), np.mean(y_prob[-int(20 * self.iti / self.bin):])


def visualize(all_mice, reward, decoder):
    if reward == 'water':
        color = 'darkblue'
    else:
        color = 'darkred'
    start_test = [m[0] for m in all_mice]
    end_test = [m[1] for m in all_mice]

    plt.figure(figsize=(10, 10))
    plt.title(f'Satiation decoder {reward}')
    plt.bar(['start', 'end'], [np.mean(start_test), np.mean(end_test)], color=color, alpha=0.6)
    for mouse in all_mice:
        plt.plot(['start', 'end'], [mouse[0], mouse[1]], color='gray', alpha=0.4)
    plt.ylim(0, 1)

    plt.savefig(join(RESULTS_PATH, decoder, 'satiation', f'summary_satiation_{decoder}_{reward}.jpg'))
    plt.close()


def main():
    # for decoder_type in ['svm', 'logistic_regression', 'gaussian', 'lda']:
    for decoder_type in ['random_forest', 'neural_network']:
        for reward in ['water', 'food']:
            week = f'opto_{reward}_week'
            days = f'opto_{reward}_days'
            path = join(RESULTS_PATH, decoder_type, 'satiation')
            all_mice = []
            for mouse_dict in MICE:
                if days in mouse_dict.keys():
                    mouse = Mouse(mouse_dict, mouse_dict[days], mouse_dict[week])
                    process = DecoderSatiation(mouse, reward, decoder_type, path)
                    all_mice.append(process.run())
                    print(f'finished mouse {mouse.name}')
            visualize(all_mice, reward, decoder_type)


'__main__' == __name__ and main()
