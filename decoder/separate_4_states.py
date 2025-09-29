import numpy as np
from os import mkdir
from os.path import join, isdir
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from configurations import *
from mouse import Mouse
from decoder.decoder_analyzer import DecoderAnalyzer

MICE = ALL_MICE
WEEK = 'week2'
DAYS = f'{WEEK}_days'
FULL_FOOD = False


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
        vectors_dict = np.load(join(self.week_data, 'axes', 'axes_vector_neutral.npy'), allow_pickle=True)[()]
        last_trial_water = vectors_dict[self.days[1].name]['last_index_vector']
        binned_water = self.decoder_analyzer.bin_data(iti_dict[self.days[1].name]['cells'], self.sec_hz)
        n_water = self.decoder_analyzer.find_satiation_period(iti_dict[self.days[1].name], last_trial_water)

        if FULL_FOOD:
            binned_food, n_food = self.food_data(iti_dict)
        else:
            binned_food = self.decoder_analyzer.bin_data(iti_dict[self.days[3].name]['cells'], self.sec_hz)
            last_trial_food = vectors_dict[self.days[3].name]['last_index_vector']
            n_food = self.decoder_analyzer.find_satiation_period(iti_dict[self.days[3].name], last_trial_food)

        chunk = int(max(n_water, n_food) * self.iti / self.bin)

        thirsty = binned_water.iloc[:, :chunk]
        quenched = binned_water.iloc[:, -chunk:]
        hungry = binned_food.iloc[:, :chunk]
        sated = binned_food.iloc[:, -chunk:]

        x_train = np.hstack([thirsty, quenched, hungry, sated]).T
        y_train = np.hstack([np.zeros(chunk), np.ones(chunk), np.full(chunk, 2), np.full(chunk, 3)])

        if self.decoder == 'svm':
            clf = SVC(kernel="linear", probability=True)
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
        clf.fit(x_train, y_train)
        return clf

    def food_data(self, iti_dict):
        day1a = iti_dict[self.days[3].name]
        day1b = iti_dict[self.days[4].name]

        binned_daya = self.decoder_analyzer.bin_data(day1a['cells'], self.sec_hz)
        binned_dayb = self.decoder_analyzer.bin_data(day1b['cells'], self.sec_hz)
        concatenated_day = pd.concat([binned_daya, binned_dayb], axis=1)

        last_ate = np.where((day1b['trials_type'] == self.days[0].ate) |
                            (day1b['trials_type'] == self.days[0].omission_food_taste))[0][-1]
        n_trials = self.decoder_analyzer.find_satiation_period(day1b, last_ate)
        return concatenated_day, n_trials

    def visualize(self, days_prob, onsets):
        labels = ['thirsty', 'quenched', 'hungry', 'sated']
        colors = ['darkblue', 'lightblue', 'darkred', 'orange']

        if FULL_FOOD:
            title_food = f'{self.mouse.name} full food prob {WEEK}'
            name_food = f'{self.mouse.name}_full_food_prob_{WEEK}.jpg'
        else:
            title_food = f'{self.mouse.name} food prob {WEEK}'
            name_food = f'{self.mouse.name}_food_prob_{WEEK}.jpg'

        plt.figure(figsize=(30, 20))
        plt.suptitle(f'{self.mouse.name} water prob {WEEK}', fontsize=20)

        for day in range(4):
            plt.subplot(2, 2, day + 1)
            plt.title(f'day {day + 1}', fontsize=20)
            onsets_water, onsets_food = onsets[day]
            for state in range(2):
                plt.plot(days_prob[day][:, state], label=labels[state], linewidth=2, color=colors[state])

            for o in onsets_water:
                plt.axvline(o, color='blue', linestyle='--', alpha=0.5, linewidth=2)
            for o in onsets_food:
                plt.axvline(o, color='red', linestyle='--', alpha=0.5, linewidth=2)

            plt.ylim(-0.05, 1.05)
            plt.legend(fontsize=15)

        plt.savefig(join(self.path, f'{self.mouse.name}_water_prob_{WEEK}.jpg'))
        plt.close()

        plt.figure(figsize=(30, 20))
        plt.suptitle(title_food, fontsize=20)

        for day in range(4):
            plt.subplot(2, 2, day + 1)
            plt.title(f'day {day + 1}', fontsize=20)
            onsets_water, onsets_food = onsets[day]
            for state in [2, 3]:
                plt.plot(days_prob[day][:, state], label=labels[state], linewidth=2, color=colors[state])

            for o in onsets_water:
                plt.axvline(o, color='blue', linestyle='--', alpha=0.5, linewidth=2)
            for o in onsets_food:
                plt.axvline(o, color='red', linestyle='--', alpha=0.5, linewidth=2)

            plt.ylim(-0.05, 1.05)
            plt.legend(fontsize=15)

        plt.savefig(join(self.path, name_food))
        plt.close()

    def run(self):
        iti_dict = np.load(join(self.week_data, 'normalized_itis.npy'), allow_pickle=True)[()]
        clf = self.train_data(iti_dict)

        prob_days = []
        onsets = []
        for i, day in enumerate(self.days[:4]):
            binned_day = self.decoder_analyzer.bin_data(iti_dict[day.name]['cells'], self.bin)
            prob_days.append(clf.predict_proba(binned_day.T))
            onsets.append(self.decoder_analyzer.blocks_onsets(iti_dict[day.name], day, self.iti // self.bin))

        self.visualize(prob_days, onsets)


def main():
    for decoder_type in ['svm', 'random_forest', 'neural_network']:
        path = join(RESULTS_PATH, decoder_type, '4_states', 'separate')
        if WEEK == 'week1':
            path = join(path, 'week1')
        if FULL_FOOD:
            path = join(path, 'full_food')
        isdir(path) or mkdir(path)
        for mouse_dict in MICE:
            if DAYS in mouse_dict.keys():
                mouse = Mouse(mouse_dict, mouse_dict[DAYS], mouse_dict[WEEK])
                process = DecoderSatiation(mouse, decoder_type, path)
                process.run()
                print(f'finished mouse {mouse.name}')


'__main__' == __name__ and main()
