import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from decoder.decoder_analyzer import DecoderAnalyzer

from configurations import *
from mouse import Mouse

MICE = ALL_MICE
WEEK = 'week2'
DAYS = f'{WEEK}_days'
PATH = join(RESULTS_PATH, 'decoder', 'mean_start_end_week')


class DecoderSatiation:

    def __init__(self, mouse):
        self.mouse = mouse
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

        clf = SVC(kernel="linear", probability=True)
        clf.fit(x_train, y_train)
        return clf

    def run(self):
        chunk = int(self.iti / self.bin) * 10
        iti_dict = np.load(join(self.week_data, 'normalized_itis.npy'), allow_pickle=True)[()]
        clf_water = self.train_data(iti_dict[self.days[1].name], self.days[1].name)
        clf_food = self.train_data(iti_dict[self.days[3].name], self.days[3].name)

        thirst = []
        hunger = []
        for i, day in enumerate(self.days[:4]):
            binned_day = self.decoder_analyzer.bin_data(iti_dict[day.name]['cells'], self.bin)
            water_prob = clf_water.predict_proba(binned_day.T)[:, 1]
            food_prob = clf_food.predict_proba(binned_day.T)[:, 1]

            if WEEK == 'week1' or i in [1, 3]:
                thirst.append([np.mean(water_prob[:chunk]), np.mean(water_prob[-chunk:])])
                hunger.append([np.mean(food_prob[:chunk]), np.mean(food_prob[-chunk:])])
            else:
                first_food = [i for i, v in enumerate(iti_dict[day.name]['trials_type'])
                              if v in [day.ate, day.not_ate, day.not_ate_licked, day.omission_food_taste]][0]
                length = len([i for i in iti_dict[day.name]['relevant_trials']
                              if i < first_food]) * (self.iti / self.bin)
                thirst.append([np.mean(water_prob[:length]), np.mean(water_prob[-length:])])
                hunger.append([np.mean(food_prob[:length]), np.mean(food_prob[-length:])])

        return thirst, hunger


def config_barplot(all_mice, need_type, colors):
    n_days = len(all_mice[0])
    x = np.arange(n_days)
    x_labels = [f'day {i + 1}' for i in range(n_days)]
    width = 0.35

    start, end = [[np.mean([m[d][i] for m in all_mice]) for d in range(n_days)] for i in [0, 1]]
    plt.bar(x - width / 2, start, width, label='Start', color=colors[0])
    plt.bar(x + width / 2, end, width, label='End', color=colors[1])
    for i in range(len(x)):
        for mouse in all_mice:
            plt.plot([x[i] - width / 2, x[i] + width / 2], mouse[i], color='gray', alpha=0.5)
    plt.xticks(x, x_labels)
    plt.ylabel(f'% {need_type}')
    plt.legend()


def visualize(all_thirst, all_hunger):
    plt.figure(figsize=(15, 15))

    plt.subplot(2, 1, 1)
    plt.title('Thirst decoder')
    config_barplot(all_thirst, 'Thirst', ['darkblue', 'lightblue'])

    plt.subplot(2, 1, 2)
    plt.title('Hunger decoder')
    config_barplot(all_hunger, 'Hunger', ['darkred', 'lightcoral'])

    plt.savefig(join(PATH, f'start_end_summary_{WEEK}.jpg'))
    plt.close()


def main():
    all_thirst, all_hunger = [], []
    for mouse_dict in MICE:
        if DAYS in mouse_dict.keys():
            mouse = Mouse(mouse_dict, mouse_dict[DAYS], mouse_dict[WEEK])
            process = DecoderSatiation(mouse)
            thirst, hunger = process.run()
            all_thirst.append(thirst)
            all_hunger.append(hunger)
            print(f'finished mouse {mouse.name}')
    visualize(all_thirst, all_hunger)


'__main__' == __name__ and main()
