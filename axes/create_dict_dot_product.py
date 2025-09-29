import numpy as np
from os.path import join

from configurations import *
from mouse import Mouse
from Processor import Processor
from axes_analyzer import AxesAnalyzer

# REWARD = 'food'
# RELEVANT_DAYS = f'opto_{REWARD}_days'
# WEEK = f'opto_{REWARD}_week'
RELEVANT_DAYS = 'week2_days'
WEEK = 'week2'
RELEVANT_MICE = ALL_MICE
VECTOR_DICT_NAME = 'axes_vector_neutral.npy'

REWARD_TRIALS = False
if REWARD_TRIALS:
    DICT_DOT_PRODUCT_NAME = 'dot_products_dict_rewards'
else:
    DICT_DOT_PRODUCT_NAME = 'dot_products_dict'


class DotProductDict(Processor):

    def __init__(self, mouse, vector_name, iti_dict_name, dict_name, *args, **kwargs):
        Processor.__init__(self, *args, **kwargs)
        self.mouse = mouse
        self.days = self.mouse.days
        self.week = mouse.week
        self.vector_name = vector_name
        self.iti_dict_name = iti_dict_name
        self.dict_name = dict_name
        self.analyzer = AxesAnalyzer(mouse.smooth_factor)

    def run(self):
        vectors_dict = np.load(join(self.week.data_dir_path, 'axes', self.vector_name), allow_pickle=True)[()]
        cells_iti_df = np.load(join(self.week.data_dir_path, 'axes', self.iti_dict_name), allow_pickle=True)[()]
        dot_products = {'vector': self.vector_name, 'iti_dict': self.iti_dict_name}

        if self.days[0].is_light:
            relevant_days = [day for day in self.days if day.name[-1] != 'b']
            vector_day1, vector_day2, vector_day3 = [vectors_dict[day.name] for day in relevant_days]

            for day in relevant_days:
                day_iti_df = cells_iti_df[day.name]['iti_df']
                day_dict = {
                    'vector_1': self.analyzer.create_dot_product_iti(day_iti_df, vector_day1),
                    'vector_2': self.analyzer.create_dot_product_iti(day_iti_df, vector_day2),
                    'vector_3': self.analyzer.create_dot_product_iti(day_iti_df, vector_day3),
                    'relevant_trials': cells_iti_df[day.name]['relevant_trials'],
                    'trials_type': cells_iti_df[day.name]['trials_classification'],
                    'last_vector_trial': vectors_dict[day.name]['last_index_vector']}
                dot_products[day.name] = day_dict

        else:
            water_day = self.days[1].name
            food_day = self.days[3].name

            for i, day in enumerate(self.days[:4]):
                day_iti_df = cells_iti_df[day.name]['iti_df']

                if day.name in vectors_dict.keys():
                    if REWARD_TRIALS:
                        last_trial = [i for i in cells_iti_df[day.name]['relevant_trials']
                                      if i < vectors_dict[day.name]['last_index_vector']][-1]
                    else:
                        last_trial = vectors_dict[day.name]['last_index_vector']
                else:
                    last_trial = cells_iti_df[day.name]['relevant_trials'][-1]
                day_dict = {
                    'water': self.analyzer.create_dot_product_iti(day_iti_df, vectors_dict[water_day]),
                    'food': self.analyzer.create_dot_product_iti(day_iti_df, vectors_dict[food_day]),
                    'food_ortho': self.analyzer.create_dot_product_iti(
                        day_iti_df, vectors_dict['food_ortho']),
                    'water_ortho': self.analyzer.create_dot_product_iti(
                        day_iti_df, vectors_dict['water_ortho']),
                    'relevant_trials': cells_iti_df[day.name]['relevant_trials'],
                    'trials_type': cells_iti_df[day.name]['trials_classification'],
                    'last_vector_trial': last_trial}
                dot_products[day.name] = day_dict

        np.save(join(self.week.data_dir_path, 'axes', f'{self.mouse.name}_{self.dict_name}.npy'), dot_products)


def main():
    for mouse_p in RELEVANT_MICE:
        if WEEK in mouse_p.keys():
            mouse = Mouse(mouse_p, mouse_p[RELEVANT_DAYS], mouse_p[WEEK])
            if REWARD_TRIALS:
                iti_dict_name = f'{mouse.name}_iti_dict_rewards.npy'
            else:
                iti_dict_name = f'{mouse.name}_iti_dict.npy'
            process = DotProductDict(
                mouse, vector_name=VECTOR_DICT_NAME, iti_dict_name=iti_dict_name, dict_name=DICT_DOT_PRODUCT_NAME)
            process.run()
            print(f'finished processing {mouse.name}')


'__main__' == __name__ and main()
