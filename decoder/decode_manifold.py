import random
import numpy as np
from os.path import join
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from configurations import *
from mouse import Mouse
from decoder.decoder_analyzer import DecoderAnalyzer
from manifold import internal

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
        bin_value = 6
        self.trial_size = bin_value * 6
        self.day_i = dai_i
        self.decoder_type = decoder_type
        self.analyzer = DecoderAnalyzer(self.sec_hz)

    def train_test(self, trials_blocks, data_trials):
        water_trials, food_trials = trials_blocks
        len_water = int(len(water_trials) * 0.8)
        len_food = int(len(food_trials) * 0.8)

        water_success = []
        food_success = []
        for _ in range(500):
            water_selection = sorted(random.sample(water_trials, len_water))
            food_selection = sorted(random.sample(food_trials, len_food))
            full_water = np.concatenate([data_trials[t] for t in water_selection], axis=1)
            test_water = np.concatenate([data_trials[t] for t in water_trials if t not in water_selection], axis=1)
            full_food = np.concatenate([data_trials[t] for t in food_selection], axis=1)
            test_food = np.concatenate([data_trials[t] for t in food_trials if t not in food_selection], axis=1)

            x_train = np.concatenate([full_water, full_food], axis=1).T
            y_train = np.hstack([
                np.ones(full_water.shape[1]),
                np.zeros(full_food.shape[1])
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

            water_pred = sum(clf.predict(test_water.T)) / test_water.shape[1] * 100
            food_pred = (1 - sum(clf.predict(test_food.T)) / test_food.shape[1]) * 100
            water_success.append(water_pred)
            food_success.append(food_pred)

        return np.mean(water_success), np.mean(food_success)

    def identify_blocks(self, iti_dict):
        day = self.days[self.day_i]

        relevant_trials = iti_dict['relevant_trials']
        trials_type = iti_dict['trials_type']
        food_trials = []
        water_trials = [0]

        current = 'water'
        for order_i, i in enumerate(relevant_trials[1:]):
            if trials_type[i - 1] in [
                    day.ate, day.not_ate, day.omission_pellet, day.not_ate_licked, day.omission_food_taste]:
                current = 'food'
            elif trials_type[i - 1] in [day.drank, day.not_drank, day.omission_water]:
                current = 'water'

            if current == 'water':
                water_trials.append(order_i + 1)
            else:
                food_trials.append(order_i + 1)

        return water_trials, food_trials

    def load_dataset_trials(self):
        data_name = f'{self.mouse.name}_week2_day{self.day_i + 1}'
        internal._set_base_directory(join(MANIFOLD_PATH, f'week2_day{self.day_i + 1}'))
        dataset = internal.get_dataset(data_name, alias="lem_final")
        data_by_trials = [dataset[:, i * self.trial_size:(i + 1) * self.trial_size]
                          for i in range(dataset.shape[1] // self.trial_size)]
        return data_by_trials

    def run(self, iti_dict):
        trials_blocks = self.identify_blocks(iti_dict)
        dataset_trials = self.load_dataset_trials()
        water_perc, food_perc = self.train_test(trials_blocks, dataset_trials)
        print(f'{self.mouse.name} day {self.day_i + 1} water: {water_perc:.1f}%, food: {food_perc:.1f}%')
        return water_perc, food_perc


def visualize(water_perc, food_perc):
    water_day1 = [wp[0] for wp in water_perc]
    water_day3 = [wp[1] for wp in water_perc]
    food_day1 = [fp[0] for fp in food_perc]
    food_day3 = [fp[1] for fp in food_perc]

    titles = ['Day 1', 'Day 3']

    plt.figure(figsize=(20, 20))
    plt.suptitle('Decoding performance within mixed days', fontsize=30)

    for i_day, days in enumerate([[water_day1, food_day1], [water_day3, food_day3]]):
        water, food = days
        plt.subplot(1, 2, i_day + 1)
        plt.title(titles[i_day], fontsize=20)
        plt.bar(['water', 'food'], [np.mean(water), np.mean(food)], color=['darkblue', 'darkred'], alpha=0.4)
        for i in range(len(water)):
            plt.plot(['water', 'food'], [water[i], food[i]], color='gray', alpha=0.4)
        plt.ylim(0, 100)
        plt.ylabel('Decoding accuracy (%)', fontsize=15)

    plt.savefig(join(RESULTS_PATH, 'logistic_regression', 'mixed_days', 'summary_mixed_days_manifold_decoder.jpg'))


def main():
    for decoder_type in ['logistic_regression']:
        water_perc, food_perc = [], []
        for mouse_dict in MICE:
            if DAYS in mouse_dict.keys():
                path = join(RESULTS_PATH, decoder_type, 'mixed_days')
                mouse = Mouse(mouse_dict, mouse_dict[DAYS], mouse_dict[WEEK])
                iti_dict = np.load(
                    join(mouse.week.data_dir_path, 'axes', f'{mouse.name}_dot_products_dict.npy'),
                    allow_pickle=True)[()]
                mouse_perc_water, mouse_perc_food = [], []
                for dai_i in [0, 2]:
                    process = MixedDays(mouse, dai_i, decoder_type, path)
                    perc_succ = process.run(iti_dict[mouse.days[dai_i].name])
                    mouse_perc_water.append(perc_succ[0])
                    mouse_perc_food.append(perc_succ[1])
                water_perc.append(mouse_perc_water)
                food_perc.append(mouse_perc_food)
                print(f'finished mouse {mouse.name}')
        visualize(water_perc, food_perc)


'__main__' == __name__ and main()
