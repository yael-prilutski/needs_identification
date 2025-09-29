import numpy as np
import matplotlib.pyplot as plt
from os.path import join

from configurations import *
from mouse import Mouse
from Processor import Processor


RELEVANT_MICE = ALL_MICE
WEEK = 'week1'
DAYS = f'{WEEK}_days'
DICT_NAME = 'dot_products_dict.npy'
PATH = join(RESULTS_PATH, 'axes', 'state_identification', 'start_end_summary')


class StartEndSummary(Processor):

    def __init__(self, mouse, dict_name, *args, **kwargs):
        Processor.__init__(self, *args, **kwargs)
        self.mouse = mouse
        self.days = mouse.days
        self.week = mouse.week
        self.sec_hz = mouse.sec_hz
        self.iti_size = self.sec_hz * 3
        self.dict_path = join(self.week.data_dir_path, 'axes', f'{self.mouse.name}_{dict_name}')

    def run(self):
        chunk = self.iti_size * 10
        products_dict = np.load(self.dict_path, allow_pickle=True)[()]

        thirst = []
        hunger = []

        for i, day in enumerate(self.days[:4]):
            last_trial = products_dict[day.name]['last_vector_trial']
            length_day = np.where(products_dict[day.name]['relevant_trials'] == last_trial)[0][0] * self.iti_size
            axis_thirst = products_dict[day.name]['water_ortho'][:length_day]
            axis_hunger = products_dict[day.name]['food_ortho'][:length_day]
            if WEEK == 'week1' or i in [1, 3]:
                thirst.append([np.mean(axis_thirst[:chunk]), np.mean(axis_thirst[-chunk:])])
                hunger.append([np.mean(axis_hunger[:chunk]), np.mean(axis_hunger[-chunk:])])
            else:
                first_food = [i for i, v in enumerate(products_dict[day.name]['trials_type'])
                              if v in [day.ate, day.not_ate, day.not_ate_licked, day.omission_food_taste]][0]
                length = len([i for i in products_dict[day.name]['relevant_trials'] if i < first_food]) * self.iti_size
                thirst.append([np.mean(axis_thirst[:length]), np.mean(axis_thirst[-length:])])
                hunger.append([np.mean(axis_hunger[:length]), np.mean(axis_hunger[-length:])])
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
    plt.ylabel(f'Location on {need_type}')
    plt.legend()


def visualize(all_thirst, all_hunger):
    plt.figure(figsize=(15, 15))

    plt.subplot(2, 1, 1)
    plt.title('Thirst axis')
    config_barplot(all_thirst, 'Thirst', ['darkblue', 'lightblue'])

    plt.subplot(2, 1, 2)
    plt.title('Hunger axis')
    config_barplot(all_hunger, 'Hunger', ['darkred', 'lightcoral'])

    plt.savefig(join(PATH, f'ortho_start_end_summary_{WEEK}.jpg'))
    plt.close()


def main():
    all_thirst, all_hunger = [], []
    for mouse_p in RELEVANT_MICE:
        if WEEK in mouse_p.keys():
            mouse = Mouse(mouse_p, mouse_p[DAYS], mouse_p[WEEK])
            process = StartEndSummary(mouse, dict_name=DICT_NAME)
            thirst, hunger = process.run()
            all_thirst.append(thirst)
            all_hunger.append(hunger)
            print(f'finished processing {mouse.name}')
    visualize(all_thirst, all_hunger)


'__main__' == __name__ and main()
