import numpy as np
import matplotlib.pyplot as plt

from configurations import *
from mouse import Mouse
from decoder.decoder_analyzer import DecoderAnalyzer
from manifold import internal

MICE = ALL_MICE
WEEK = 'week2'
DAYS = f'{WEEK}_days'
PATH = join(RESULTS_PATH, 'manifold', '1D_plots')


class MixedDays:

    def __init__(self, mouse):
        self.mouse = mouse
        self.days = mouse.days
        self.week_data = mouse.week.data_dir_path
        self.sec_hz = mouse.sec_hz
        bin_value = 6
        self.trial_size = bin_value * 6
        self.analyzer = DecoderAnalyzer(self.sec_hz)

    def identify_blocks(self, iti_dict, day):
        relevant_trials = iti_dict['relevant_trials']
        trials_type = iti_dict['trials_type']
        food_onsets = []
        water_onsets = [0]

        current = 'water'
        for order_i, i in enumerate(relevant_trials[1:]):
            if current == 'water' and trials_type[i - 1] in [
                    day.ate, day.not_ate, day.omission_pellet, day.not_ate_licked, day.omission_food_taste]:
                current = 'food'
                food_onsets.append((order_i + 1) * self.trial_size)

            elif current == 'food' and trials_type[i - 1] in [day.drank, day.not_drank, day.omission_water]:
                current = 'water'
                water_onsets.append((order_i + 1) * self.trial_size)

        return water_onsets, food_onsets

    def load_dataset(self, day_i):
        data_name = f'{self.mouse.name}_week2_day{day_i + 1}'
        internal._set_base_directory(join(MANIFOLD_PATH, f'week2_day{day_i + 1}'))
        dataset = internal.get_dataset(data_name, alias="lem_final")
        return dataset

    def visualize(self, onsets, datasets):
        days_titles = ['Day 1', 'Day 3']
        fig, axes = plt.subplots(4, 2, figsize=(30, 30))
        plt.suptitle('manifold ', fontsize=30)
        for i_day in range(2):
            water_o, food_o = onsets[i_day]
            for i_dim in range(4):
                data = datasets[i_day][i_dim, :]
                ax = axes[i_dim, i_day]
                ax.set_title(f"{days_titles[i_day]}, Dim {i_dim}")
                ax.plot(data)
                for w in water_o:
                    ax.axvline(w, color='blue', linestyle='--', linewidth=2)
                for f in food_o:
                    ax.axvline(f, color='red', linestyle='--', linewidth=2)
        plt.savefig(join(PATH, f'{self.mouse.name}_mixed_days.jpg'))
        plt.close()

    def run(self):
        iti_dict = np.load(
            join(self.week_data, 'axes', f'{self.mouse.name}_dot_products_dict.npy'), allow_pickle=True)[()]

        two_days_onsets = []
        two_days_data = []
        for i in [0, 2]:
            day = self.days[i]
            two_days_data.append(self.load_dataset(i))
            two_days_onsets.append(self.identify_blocks(iti_dict[day.name], day))
        self.visualize(two_days_onsets, two_days_data)


def main():
    for mouse_dict in MICE:
        if DAYS in mouse_dict.keys():
            mouse = Mouse(mouse_dict, mouse_dict[DAYS], mouse_dict[WEEK])
            process = MixedDays(mouse)
            process.run()


'__main__' == __name__ and main()
