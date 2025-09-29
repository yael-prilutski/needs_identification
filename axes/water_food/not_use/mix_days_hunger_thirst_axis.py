import numpy as np
from os import makedirs
from os.path import join
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

from configurations import *
from mouse import Mouse
from Processor import Processor
from axes.axes_analyzer import AxesAnalyzer

RELEVANT_DAYS = 'week2_days'
WEEK = 'week2'
RELEVANT_MICE = ALL_MICE
VECTOR_DICT_NAME = 'axes_vector_neutral.npy'
PATH = join(RESULTS_PATH, 'axes', 'mix_days_hunger_thirst_axis')
SMOOTH = True
if not SMOOTH:
    PATH = join(PATH, 'no_smooth')
makedirs(PATH, exist_ok=True)


class DotProductDict(Processor):

    def __init__(self, mouse, vector_name, iti_dict_name, *args, **kwargs):
        Processor.__init__(self, *args, **kwargs)
        self.mouse = mouse
        self.days = self.mouse.days
        self.week = mouse.week
        self.vector_name = vector_name
        self.iti_dict_name = iti_dict_name
        self.analyzer = AxesAnalyzer(mouse.smooth_factor)
        self.iti_size = self.analyzer.iti_slice
        self.flat_parameter = int(3 * mouse.smooth_factor)

    def visualize(self, dot_products, colors):
        plt.figure(figsize=[27, 18])
        plt.suptitle(f'{self.mouse.name} mix days hunger thirst axis- trials', fontsize=30)

        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.scatter(np.arange(len(dot_products[i])), dot_products[i], c=colors[i], edgecolors='None')
            plt.title(f'day {i + 1}', fontsize=25)
            plt.ylabel('1 - thirst\n0 - hunger', fontsize=20)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.ylim(min(0, min(dot_products[i]) - 0.1), max(1, max(dot_products[i]) + 0.1))
        plt.savefig(join(PATH, f'{self.mouse.name}_hunger_thirst_axis.jpg'))
        plt.close()

    def create_hunger_thirst_axis(self, vectors_dict):
        thirst = vectors_dict[self.days[1].name]['start_population']
        hunger = vectors_dict[self.days[3].name]['start_population']
        sub_thirst_hunger = hunger - thirst
        vector_dict = {
            'sub': sub_thirst_hunger,
            'start': np.dot(thirst, sub_thirst_hunger),
            'end': np.dot(hunger, sub_thirst_hunger)}
        return vector_dict

    def run(self):
        vectors_dict = np.load(join(self.week.data_dir_path, 'axes', self.vector_name), allow_pickle=True)[()]
        thirst_hunger_dict = self.create_hunger_thirst_axis(vectors_dict)
        cells_iti_df = np.load(join(self.week.data_dir_path, 'axes', self.iti_dict_name), allow_pickle=True)[()]

        dot_products = []
        colors = []
        for i, day in enumerate(self.days[:4]):
            day_iti_df = cells_iti_df[day.name]['iti_df']
            if day.name in vectors_dict.keys():
                last_trial = vectors_dict[day.name]['last_index_vector']
            else:
                last_trial = cells_iti_df[day.name]['relevant_trials'][-1]
            temp_dict = {
                'last_vector_trial': last_trial,
                'relevant_trials': cells_iti_df[day.name]['relevant_trials'],
                'trials_type': cells_iti_df[day.name]['trials_classification']
            }
            length_day = np.where(cells_iti_df[day.name]['relevant_trials'] == last_trial)[0][0] * self.iti_size
            dot_product = self.analyzer.create_dot_product_iti(day_iti_df, thirst_hunger_dict)[:length_day]
            day_colors = self.analyzer.classify_by_behavior_colors(day, temp_dict, length_day, self.iti_size)
            if SMOOTH:
                dot_product = uniform_filter1d(dot_product, size=self.flat_parameter)
            dot_products.append(dot_product[:length_day])
            colors.append(day_colors)
        self.visualize(dot_products, colors)


def main():
    for mouse_p in RELEVANT_MICE:
        if WEEK in mouse_p.keys():
            mouse = Mouse(mouse_p, mouse_p[RELEVANT_DAYS], mouse_p[WEEK])
            iti_dict_name = f'{mouse.name}_iti_dict.npy'
            process = DotProductDict(mouse, vector_name=VECTOR_DICT_NAME, iti_dict_name=iti_dict_name)
            process.run()
            print(f'finished processing {mouse.name}')


'__main__' == __name__ and main()
