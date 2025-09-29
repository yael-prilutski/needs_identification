import numpy as np
from os.path import join
from scipy import stats
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

from configurations import *
from mouse import Mouse
from run import Run

MICE = ALL_MICE
DAYS = 'week1_days'
MIN_TRIALS = 20


class AnticipatoryLicks:

    def __init__(self, mouse):
        self.mouse = mouse
        self.days = self.mouse.days
        self.sec_hz = mouse.smooth_factor
        self.min_trials = MIN_TRIALS

    def find_licks(self, lick_trials, pellet_y, food_t=False):
        if len(lick_trials):
            licks_per_trial = []
            for i_trial, trial in enumerate(lick_trials):
                if food_t:
                    end = pellet_y[i_trial]
                else:
                    end = self.sec_hz * 4
                relevant_licks = np.where(np.array(trial[self.sec_hz * 3: end]) == 1)[0]
                separate_licks = [i for i in relevant_licks if i - 1 not in relevant_licks]
                final_number = len(separate_licks)
                if final_number != 0:
                    final_number = final_number / ((end - self.sec_hz * 3) / self.sec_hz)
                licks_per_trial.append(final_number)

            return np.mean(licks_per_trial)
        return 0

    def load_licks(self, day, special_day=None):
        data = day.load_data_dict()
        trials_type = data['trials_classification']
        water_licks = data['water_licks']
        food_licks = data['pellet_licks']
        y_location = data['pellet_y']
        y_trials = []

        index_drank = np.where(trials_type == day.drank)[0][:self.min_trials]
        index_ate = np.where((trials_type == day.ate) | (trials_type == day.omission_food_taste))[0][:self.min_trials]
        index_neutral = np.where(trials_type == day.neutral)[0][:self.min_trials]
        if special_day == 'water':
            return self.find_licks([water_licks[i] for i in index_drank], y_trials)

        if len(index_ate):
            end_licks = [max(np.where(np.array(y_location[t]) > 0)[0][0], self.sec_hz * 4) for t in index_ate]
        else:
            end_licks = [self.sec_hz * 4] * len(index_ate)

        if special_day == 'food':
            return self.find_licks([food_licks[i] for i in index_ate], end_licks, True)

        all_licks = []

        for i, index in enumerate([index_drank, index_ate, index_neutral]):
            licks_water = self.find_licks([water_licks[t] for t in index], y_trials)
            if i == 1:
                licks_phase = end_licks
            else:
                licks_phase = [int(np.median(end_licks))] * len(index)
            licks_pellet = self.find_licks([food_licks[i] for i in index], licks_phase, True)
            all_licks.append([licks_water, licks_pellet])
        return all_licks

    def normalize_licks(self, water_licks, food_licks, neutral_licks, water_rate, food_rate):
        water_in_water = water_licks[0] / water_rate if water_licks[0] or water_licks[0] == 0 else None
        food_in_water = water_licks[1] / food_rate if water_licks[1] or water_licks[1] == 0 else None
        water_in_food = food_licks[0] / water_rate if food_licks[0] or food_licks[0] == 0 else None
        food_in_food = food_licks[1] / food_rate if food_licks[1] or food_licks[1] == 0 else None
        water_in_neutral = neutral_licks[0] / water_rate if neutral_licks[0] or neutral_licks[0] == 0 else None
        food_in_neutral = neutral_licks[1] / food_rate if neutral_licks[1] or neutral_licks[1] == 0 else None
        return [water_in_water, food_in_water], [water_in_food, food_in_food], [water_in_neutral, food_in_neutral]

    def run(self):
        max_water_rate = self.load_licks(self.days[1], 'water')
        max_food_rate = self.load_licks(self.days[3], 'food')
        licks_rate = []
        for day in [self.days[0], self.days[2]]:
            water_licks, food_licks, neutral_licks = self.load_licks(day)
            day_runs = day.runs_paths
            if sum([Run(path, self.sec_hz).no_pellet_licks for path in day_runs]) \
                    or sum([Run(path, self.sec_hz).constant_pellet_licks for path in day_runs]):
                water_licks[1] = None
                food_licks[1] = None
                neutral_licks[1] = None
            licks_rate.append(
                self.normalize_licks(water_licks, food_licks, neutral_licks, max_water_rate, max_food_rate))
        print(licks_rate)
        return licks_rate


def visualize_all_mice(all_mice, day_type):
    water_trials = [m[0] for m in all_mice]
    food_trials = [m[1] for m in all_mice]
    neutral_trials = [m[2] for m in all_mice]

    titles = ['water trials', 'food trials', 'neutral trials']
    x_labels = ['water licks', 'food licks']
    colors = ['darkblue', 'darkred']
    plt.figure(figsize=(30, 20))
    plt.suptitle(f'Summary anticipatory licking {day_type}', fontsize=30)

    for i, trial in enumerate([water_trials, food_trials, neutral_trials]):
        water_licks = [m[0] for m in trial if m[0] is not None]
        food_licks = [m[1] for m in trial if m[1] is not None]
        relevant_licks = [i for i in range(len(trial)) if trial[i][0] is not None and trial[i][1] is not None]
        p_value = stats.ttest_ind([trial[i][0] for i in relevant_licks], [trial[i][1] for i in relevant_licks])[1]

        plt.subplot(1, 3, i + 1)
        plt.title(f'{titles[i]}, p_value: {round(p_value, 3)}', fontsize=25)

        plt.bar(x_labels, [np.mean(water_licks), np.mean(food_licks)], color=colors)
        for m in trial:
            plt.plot(x_labels, m, c='gray', alpha=0.3)
        plt.ylim([-0.1, 1.4])
        plt.ylabel('licks rate (normalized to reward-lick in the relevant day)', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

    # plt.show()
    plt.savefig(join(RESULTS_PATH, 'behavior', 'anticipatory_licking', f'{day_type}_summary_anticipatory_lickingv2.jpg'))


def main():
    thirst_day = []
    hunger_day = []
    for mouse_dict in MICE:
        if DAYS in mouse_dict.keys():
            mouse = Mouse(mouse_dict, mouse_dict[DAYS])
            processor = AnticipatoryLicks(mouse)
            thirst, hunger = processor.run()
            thirst_day.append(thirst)
            hunger_day.append(hunger)
            print(f'{mouse.name} done')
    visualize_all_mice(thirst_day, 'thirst')
    visualize_all_mice(hunger_day, 'hunger')


'__main__' == __name__ and main()
