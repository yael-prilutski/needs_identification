import numpy as np
from os import mkdir
from os.path import join, isdir
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from decoder.decoder_analyzer import DecoderAnalyzer

from configurations import *
from mouse import Mouse

MICE = ALL_MICE
WEEK = 'week1'
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

    def train_data(self, day, day_name):
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

    def visualize(self, water_days, food_days):
        if FULL_FOOD:
            title = f'{self.mouse.name} start {WEEK} full food'
            name = f'{self.mouse.name}_start_{WEEK}_full_food.jpg'
        else:
            title = f'{self.mouse.name} start {WEEK}'
            name = f'{self.mouse.name}_start_{WEEK}.jpg'

        if WEEK == 'week1':
            names = ['day1', 'day2', 'day3', 'day4']
        else:
            names = ['day1 water', 'day1 food', 'day2', 'day3 water', 'day3 food', 'day4']

        plt.figure(figsize=(20, 20))
        plt.suptitle(title, fontsize=20)

        plt.bar(np.arange(len(names)) - 0.2, water_days, width=0.4, label='water', color='lightskyblue')
        plt.bar(np.arange(len(names)) + 0.2, food_days, width=0.4, label='food', color='lightcoral')
        plt.xticks(np.arange(len(names)), names, fontsize=15)

        plt.savefig(join(self.path, name))
        plt.close()


    def run(self):
        iti_dict = np.load(join(self.week_data, 'normalized_itis.npy'), allow_pickle=True)[()]
        clf_water = self.train_data(iti_dict[self.days[1].name], self.days[1].name)
        if FULL_FOOD:
            clf_food = self.train_data_food(iti_dict)
        else:
            clf_food = self.train_data(iti_dict[self.days[3].name], self.days[3].name)

        mean_days = []
        for i, day in enumerate(self.days[:4]):
            binned_day = self.decoder_analyzer.bin_data(iti_dict[day.name]['cells'], self.bin)
            water_prob = clf_water.predict_proba(binned_day.T)[:, 1]
            food_prob = clf_food.predict_proba(binned_day.T)[:, 1]

            if i in [1, 3] or WEEK == 'week1':
                chunk = int(self.iti / self.bin) * 20
                mean_water = np.mean(water_prob[:chunk])
                mean_food = np.mean(food_prob[:chunk])
                mean_days.append([mean_water, mean_food])
            else:
                onsets_water, onsets_food = self.decoder_analyzer.blocks_onsets(iti_dict[day.name], day, self.iti // self.bin)

                block_water_w = np.mean(water_prob[: onsets_food[0]])
                block_food_w = np.mean(food_prob[: onsets_food[0]])
                mean_days.append([block_water_w, block_food_w])

                block_food_w = np.mean(water_prob[onsets_food[0]: onsets_water[1]])
                block_food_f = np.mean(food_prob[onsets_food[0]: onsets_water[1]])
                mean_days.append([block_food_w, block_food_f])

        water_days = [day[0] for day in mean_days]
        food_days = [day[1] for day in mean_days]
        self.visualize(water_days, food_days)
        return water_days, food_days


def mice_visualize(mice, path):
    if FULL_FOOD:
        title = f'Summary start {WEEK} full food'
        name = f'summary_start_{WEEK}_full_food.jpg'
    else:
        title = f'Summary start {WEEK}'
        name = f'summary_start_{WEEK}.jpg'

    if WEEK == 'week1':
        names = ['day1', 'day2', 'day3', 'day4']
    else:
        names = ['day1 water', 'day1 food', 'day2', 'day3 water', 'day3 food', 'day4']

    water_probabilities = [mouse[0] for mouse in mice]
    water_days = [[m[d] for m in water_probabilities] for d in range(len(names))]
    food_probabilities = [mouse[1] for mouse in mice]
    food_days = [[m[d] for m in food_probabilities] for d in range(len(names))]

    all_locations = sorted(np.hstack([np.arange(len(names)) + 0.2, np.arange(len(names)) - 0.2]))

    plt.figure(figsize=(20, 20))
    plt.suptitle(title, fontsize=20)

    plt.bar(np.arange(len(names)) - 0.2, np.mean(water_days, axis=1), width=0.4, label='water', color='lightskyblue')
    plt.bar(np.arange(len(names)) + 0.2, np.mean(food_days, axis=1), width=0.4, label='food', color='lightcoral')
    for i in range(len(names)):
        plt.scatter(np.ones(len(water_days[i])) * i - 0.2, water_days[i], color='black', s=50)
        plt.scatter(np.ones(len(food_days[i])) * i + 0.2, food_days[i], color='black', s=50)
    for mouse in mice:
        mice_locations = np.hstack([[mouse[0][d], mouse[1][d]] for d in range(len(names))])
        plt.plot(all_locations, mice_locations, color='gray', alpha=0.5)
    plt.xticks(np.arange(len(names)), names, fontsize=15)

    plt.savefig(join(path, name))
    plt.close()


def main():
    for decoder_type in ['svm']:
        all_mice = []
        path = join(RESULTS_PATH, decoder_type, 'mean_start_days')
        if WEEK == 'week1':
            path = join(path, 'week1')
        if FULL_FOOD:
            path = join(path, 'full_food')
        isdir(path) or mkdir(path)
        for mouse_dict in MICE:
            if DAYS in mouse_dict.keys():
                mouse = Mouse(mouse_dict, mouse_dict[DAYS], mouse_dict[WEEK])
                process = DecoderSatiation(mouse, decoder_type, path)
                all_mice.append(process.run())
                print(f'finished mouse {mouse.name}')
        mice_visualize(all_mice, path)


'__main__' == __name__ and main()
