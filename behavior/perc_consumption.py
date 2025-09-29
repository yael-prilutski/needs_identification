import numpy as np
from scipy import stats
from os.path import join
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42

from configurations import *
from mouse import Mouse
from Processor import Processor

MICE = ALL_MICE
DAYS = 'week1_days'


class PercBehavior(Processor):

    def __init__(self, mouse, *args, **kwargs):
        Processor.__init__(self, *args, **kwargs)
        self.days = mouse.days

    def analyze_day(self, day):
        trials_type = day.load_data_dict()['trials_classification']

        perc_drank_trials = 0
        perc_ate_trials = 0

        drank = len(np.where(trials_type == day.drank)[0])
        water = len(np.where((trials_type == day.drank) | (trials_type == day.not_drank))[0])
        ate = len(np.where((trials_type == day.ate) | (trials_type == day.omission_food_taste))[0])
        food = len(
            np.where((trials_type == day.ate) | (trials_type == day.omission_food_taste) | (
                    trials_type == day.not_ate) | (trials_type == day.not_ate_licked))[0])

        if drank > 0 and water > 0:
            perc_drank_trials = (drank / water) * 100
        if ate > 0 and food > 0:
            perc_ate_trials = (ate / food) * 100

        # print(f'{day.name} drank: {perc_drank_trials}, ate: {perc_ate_trials}')
        return perc_drank_trials, perc_ate_trials

    def run(self):
        return [self.analyze_day(day) for day in [self.days[0], self.days[2]]]


def visualize(all_mice):
    x_labels = ['water', 'food']
    colors = ['darkblue', 'darkred']
    titles = ['Thirsty', 'Hungry']
    plt.figure(figsize=(20, 20))
    plt.suptitle('Summary percentage consumption', fontsize=30)
    for day in range(2):
        relevant_day_values = [m[day] for m in all_mice]
        mice_by_condition = [[m[c] for m in relevant_day_values] for c in range(len(x_labels))]
        mean_values = [np.nanmean(c) for c in mice_by_condition]
        _, p_value = stats.ttest_rel(mice_by_condition[0], mice_by_condition[1])

        plt.subplot(1, 2, day + 1)
        plt.title(f'{titles[day]}, p_value: {round(p_value, 3)}', fontsize=25)
        plt.bar(x_labels, mean_values, color=colors)
        for m in relevant_day_values:
            plt.plot(x_labels, m, c='gray', alpha=0.3)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

    plt.savefig(join(RESULTS_PATH, 'behavior', 'perc_consumption', 'summary_perc_consumption.svg'))


def main():
    all_perc = []
    for mouse_dict in MICE:
        if DAYS in mouse_dict.keys():
            mouse = Mouse(mouse_dict, mouse_dict[DAYS])
            process = PercBehavior(mouse)
            all_perc.append(process.run())
            print(f'finished {mouse.name}')
    visualize(all_perc)


'__main__' == __name__ and main()
