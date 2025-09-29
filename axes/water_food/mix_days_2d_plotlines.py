import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from matplotlib.colors import LinearSegmentedColormap
from os.path import join
from scipy.ndimage import uniform_filter1d

from configurations import *
from mouse import Mouse
from Processor import Processor
from axes.axes_analyzer import AxesAnalyzer


RELEVANT_MICE =ALL_MICE
NO_ORTHO = False
WEEK = 'week2'
DAYS = 'week2_days'
DICT_NAME = 'dot_products_dict.npy'
PATH = join(RESULTS_PATH, 'axes', 'mix_days_2d')

if NO_ORTHO:
    PATH = join(PATH, 'no_ortho')


class MixDays2D(Processor):

    def __init__(self, mouse, dict_name, *args, **kwargs):
        Processor.__init__(self, *args, **kwargs)
        self.mouse = mouse
        self.days = mouse.days
        self.flat_parameter = int(10 * mouse.smooth_factor)
        self.iti_size = AxesAnalyzer(mouse.smooth_factor).iti_slice
        self.dict_path = join(mouse.week.data_dir_path, 'axes', f'{self.mouse.name}_{dict_name}')

    def truncate_colormap(self, cmap, minval=0.1, maxval=0.9, n=100):
        new_cmap = cmap(np.linspace(minval, maxval, n))
        return lambda x: new_cmap[(x * (n - 1)).astype(int)]

    def visualize(self, dot_products):
        titles = ['day1', 'day3']

        plt.figure(figsize=[24, 12])
        plt.suptitle(f'Mix days 2d trials- {self.mouse.name}', fontsize=30)

        for i in range(2):
            thirst, hunger = dot_products[i]
            min_value = min(0, min(thirst) - 0.1, min(hunger) - 0.1)
            max_value = max(1, max(thirst) + 0.1, max(hunger) + 0.1)
            time = np.linspace(0, 1, len(thirst))
            gray_map = self.truncate_colormap(plt.cm.Greys, 0.2, 0.8)
            colors = gray_map(time)

            plt.subplot(1, 2, i + 1)
            plt.title(titles[i], fontsize=20)
            plt.scatter(thirst, hunger, c=colors, edgecolors='None', zorder=0, rasterized=True)
            plt.gca().set_rasterization_zorder(1)

            new_cmap = LinearSegmentedColormap.from_list("trunc_greys", plt.cm.Greys(np.linspace(0.2, 0.8, 256)))
            sm = plt.cm.ScalarMappable(cmap=new_cmap, norm=plt.Normalize(0, 1))
            plt.colorbar(sm, label="Time progression")

            plt.xlabel('thirst axis', fontsize=20)
            plt.ylabel('hunger axis', fontsize=20)
            plt.xlim(min_value, max_value)
            plt.ylim(min_value, max_value)
            plt.legend()
            plt.gca().set_aspect('equal', adjustable='box')

        # plt.show()
        plt.savefig(join(PATH, f'{self.mouse.name}_mix_days_2d.jpg'))
        plt.close()

    def run(self):
        products_dict = np.load(self.dict_path, allow_pickle=True)[()]
        relevant_days = [self.days[0], self.days[2]]

        dot_products = []
        for i, day in enumerate(relevant_days):
            last_trial = products_dict[day.name]['last_vector_trial']
            length_day = np.where(products_dict[day.name]['relevant_trials'] == last_trial)[0][0] * self.iti_size
            if NO_ORTHO:
                axis_thirst = products_dict[day.name]['water'][:length_day]
                axis_hunger = products_dict[day.name]['food'][:length_day]
            else:
                axis_thirst = products_dict[day.name]['water_ortho'][:length_day]
                axis_hunger = products_dict[day.name]['food_ortho'][:length_day]
            dot_products.append(
                [uniform_filter1d(axis_thirst, self.flat_parameter),
                 uniform_filter1d(axis_hunger, self.flat_parameter)])

        self.visualize(dot_products)
        return dot_products


def main():
    for mouse_p in RELEVANT_MICE:
        if WEEK in mouse_p.keys():
            mouse = Mouse(mouse_p, mouse_p[DAYS], mouse_p[WEEK])
            process = MixDays2D(mouse, dict_name=DICT_NAME)
            process.run()
            print(f'finished processing {mouse.name}')


'__main__' == __name__ and main()
