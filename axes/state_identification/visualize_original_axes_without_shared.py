import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

from configurations import *
from mouse import Mouse
from Processor import Processor
from axes.axes_analyzer import AxesAnalyzer

WEEK = 'week1'
RELEVANT_DAYS = f'{WEEK}_days'
RELEVANT_MICE = ALL_MICE


class clean_water_food_axes(Processor):

    def __init__(self, mouse, *args, **kwargs):
        Processor.__init__(self, *args, **kwargs)
        self.mouse = mouse
        self.days = mouse.days
        self.sec_hz = mouse.sec_hz
        self.chunk = self.sec_hz
        self.week = mouse.week
        self.analyzer = AxesAnalyzer(mouse.sec_hz)

    def visualize(self, water, food, orig_water, orig_food):
        plt.figure(figsize=(30, 15))
        plt.suptitle(f'{self.mouse.name} original vs cleaned axes')

        for i in range(4):
            plt.subplot(2, 4, i + 1)
            plt.title(f'Day {i + 1} cleaned axes')
            plt.plot(uniform_filter1d(water[i], self.chunk), label='water', color='darkblue', alpha=0.7, linewidth=2)
            plt.plot(uniform_filter1d(food[i], self.chunk), label='food', color='darkred', alpha=0.7, linewidth=2)
            plt.legend()

            plt.subplot(2, 4, i + 5)
            plt.title(f'Day {i + 1} original axes')
            plt.plot(uniform_filter1d(orig_water[i], self.chunk), label='water', color='darkblue', alpha=0.7, linewidth=2)
            plt.plot(uniform_filter1d(orig_food[i], self.chunk), label='food', color='darkred', alpha=0.7, linewidth=2)
            plt.legend()

        plt.savefig(join(RESULTS_PATH, 'axes', 'state_identification', 'cleaned_water_food_week',
                         f'{self.mouse.name}_{WEEK}_original_vs_cleaned.jpg'))
        plt.close()

    def run(self):
        vectors_dict = np.load(join(self.week.data_dir_path, 'axes', 'clean_axes.npy'), allow_pickle=True)[()]
        water_vector = vectors_dict['cleaned_water']
        food_vector = vectors_dict['cleaned_food']
        iti_dict = np.load(
            join(self.week.data_dir_path, 'axes', f'{self.mouse.name}_iti_dict.npy'), allow_pickle=True)[()]
        original_products = np.load(join(self.week.data_dir_path, 'axes', f'{self.mouse.name}_dot_products_dict.npy'),
                                    allow_pickle=True)[()]

        dot_water, dot_food, original_water, original_food = [], [], [], []
        for day in self.days[:4]:
            if day.name in vectors_dict.keys():
                last_trial = vectors_dict[day.name]['last_index_vector']
                relevant_trials = iti_dict[day.name]['relevant_trials']
                length_trials = np.where(relevant_trials == last_trial)[0][0]
                day_data = iti_dict[day.name]['iti_df'].iloc[:, :length_trials * self.analyzer.iti_slice]
            else:
                day_data = iti_dict[day.name]['iti_df']
            dot_water.append(self.analyzer.create_dot_product_iti(day_data, water_vector))
            dot_food.append(self.analyzer.create_dot_product_iti(day_data, food_vector))
            original_water.append(original_products[day.name]['water'])
            original_food.append(original_products[day.name]['food'])

        self.visualize(dot_water, dot_food, original_water, original_food)


def main():
    for mouse_path in RELEVANT_MICE:
        if WEEK in mouse_path.keys():
            mouse = Mouse(mouse_path, mouse_path[RELEVANT_DAYS], mouse_path[WEEK])
            process = clean_water_food_axes(mouse)
            process.run()
            print(f'finished processing {mouse.name}')


'__main__' == __name__ and main()
