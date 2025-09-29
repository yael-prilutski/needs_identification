import numpy as np
from os.path import join
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

from configurations import *
from mouse import Mouse
from Processor import Processor

RELEVANT_DAYS = 'week1_days'
RELEVANT_MICE = [MOUSE_YP84]
SINGLE_REWARD_DAYS = True
PATH = join(RESULTS_PATH, 'behavior', 'cumulative_rewards')
if SINGLE_REWARD_DAYS:
    PATH = join(PATH, 'single_reward_days')


class CumulativeRewards(Processor):

    def __init__(self, mouse, *args, **kwargs):
        Processor.__init__(self, *args, **kwargs)
        self.mouse = mouse
        self.sec_hz = self.mouse.smooth_factor

    def visualize(self, thirsty_day, hungry_day, percentage=False):
        titles = ['Thirsty', 'Hungry']
        plt.figure(figsize=(20, 20))
        plt.suptitle(f'Cumulative rewards {self.mouse.name}', fontsize=30)

        for i_day, day in enumerate([thirsty_day, hungry_day]):
            plt.subplot(2, 1, i_day + 1)
            plt.title(titles[i_day], fontsize=25)
            water, food = day
            if percentage:
                water = water / water[-1]
                food = food / food[-1]
            plt.plot(water, label='water', color='royalblue')
            plt.plot(food, label='food', color='crimson')
            plt.xlabel('Trials', fontsize=20)
            plt.ylabel('Rewards', fontsize=20)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.legend()

        if percentage:
            plt.savefig(join(PATH, 'percentage', f'cumulative_rewards_{self.mouse.name}.jpg'))
        else:
            plt.savefig(join(PATH, f'cumulative_rewards_{self.mouse.name}.svg'))

        plt.close()

    def count_rewards(self, day):
        trials_type = day.load_data_dict()['trials_classification']

        water_rewards = []
        food_rewards = []
        for i in trials_type:
            if i == day.drank:
                water_rewards.append(1)
                food_rewards.append(0)
            if i in [day.ate, day.omission_food_taste]:
                water_rewards.append(0)
                food_rewards.append(1)
            else:
                water_rewards.append(0)
                food_rewards.append(0)
        return np.cumsum(water_rewards), np.cumsum(food_rewards)

    def run(self):
        if SINGLE_REWARD_DAYS:
            thirsty_day = self.count_rewards(self.mouse.days[1])
            hungry_day = self.count_rewards(self.mouse.days[3])
        else:
            thirsty_day = self.count_rewards(self.mouse.days[0])
            hungry_day = self.count_rewards(self.mouse.days[2])
        self.visualize(thirsty_day, hungry_day)
        self.visualize(thirsty_day, hungry_day, percentage=True)


def main():
    for mouse_dict in RELEVANT_MICE:
        if RELEVANT_DAYS in mouse_dict.keys():
            mouse = Mouse(mouse_dict, mouse_dict[RELEVANT_DAYS])
            process = CumulativeRewards(mouse)
            process.run()
            print(f'finished {mouse.name}')


'__main__' == __name__ and main()
