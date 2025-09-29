import numpy as np
import matplotlib
matplotlib.use('Agg')
from os.path import join
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

from configurations import *
from mouse import Mouse
from Visualizer import Visualizer

WEEK = 'week1'
DAYS = 'week1_days'
PATH = join(RESULTS_PATH, 'water_food', 'satiation', 'response_satiation_correlation')


class ResponseSatiationCorrelation:

    def __init__(self, mouse):
        self.mouse = mouse
        self.days = mouse.days
        self.week_path = self.mouse.week.data_dir_path
        self.sec_hz = mouse.days[0].smooth_factor
        self.week_data = mouse.week.data_dir_path
        self.visualizer = Visualizer(self.sec_hz)
        self.iti_size = self.sec_hz * 3

    def visualize(self, day_responses):
        titles = ['Day2', 'Day4']
        plt.figure(figsize=(21, 21))
        plt.suptitle(f'{self.mouse.name} response-satiation correlation', fontsize=20)
        for i in range(2):
            plt.subplot(1, 2, i + 1)
            self.visualizer.scatter_plot_config(day_responses[i], ['responses', 'satiation delta'], titles[i])
        plt.savefig(join(PATH, f'{self.mouse.name}_response_satiation_correlation.jpg'))
        plt.close()

    def find_day_data(self, responses, iti, length):
        x_values = responses
        y_values = []
        for cell in range(len(iti)):
            start = iti.iloc[cell][:self.iti_size * 30].mean()
            end = iti.iloc[cell][:length * self.iti_size][-self.iti_size * 30:].mean()
            y_values.append(end - start)
        return x_values, y_values

    def find_last_trial(self, water_trials, food_trials):
        vector_dict = np.load(join(self.week_path, 'axes', 'axes_vector_neutral.npy'), allow_pickle=True)[()]
        last_trial_water = vector_dict[self.days[1].name]['last_index_vector']
        last_trial_food = vector_dict[self.days[3].name]['last_index_vector']
        length_water = np.where(water_trials == last_trial_water)[0][0]
        length_food = np.where(food_trials == last_trial_food)[0][0]
        return length_water, length_food

    def run(self):
        iti_dict = np.load(join(self.week_path, 'axes', f'{self.mouse.name}_iti_dict.npy'), allow_pickle=True)[()]
        dict_responses = np.load(join(self.week_path, 'dict_mean_responses.npy'), allow_pickle=True)[()]
        length_water, length_food = self.find_last_trial(
            iti_dict[self.days[1].name]['relevant_trials'], iti_dict[self.days[3].name]['relevant_trials'])

        day2 = self.find_day_data(
            dict_responses['day_2']['water_reward'], iti_dict[self.days[1].name]['iti_df'], length_water)
        day_4 = self.find_day_data(
            dict_responses['day_4']['food_reward'], iti_dict[self.days[3].name]['iti_df'], length_food)
        # self.visualize([day2, day_4])
        print(f'finished data processing {self.mouse.name}')
        return np.corrcoef(day2[0], day2[1])[0, 1], np.corrcoef(day_4[0], day_4[1])[0, 1]


def visualize_mice(mice_corr):
    day2_corr = np.mean([m[0] for m in mice_corr])
    day4_corr = np.mean([m[1] for m in mice_corr])

    plt.figure(figsize=(21, 21))
    plt.suptitle('Response-Satiation correlation', fontsize=30)
    plt.bar(['Day2', 'Day4'], [day2_corr, day4_corr], color=['darkblue', 'darkred'])
    for m in mice_corr:
        plt.plot(['Day2', 'Day4'], m, color='gray', alpha=0.5)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(join(PATH, 'response_satiation_correlation.svg'))


def main():
    mice_corr = []
    for mouse_dict in ALL_MICE:
        if WEEK in mouse_dict.keys():
            mouse = Mouse(mouse_dict, mouse_dict[DAYS], mouse_dict[WEEK])
            processor = ResponseSatiationCorrelation(mouse)
            mice_corr.append(processor.run())
    visualize_mice(mice_corr)


'__main__' == __name__ and main()
