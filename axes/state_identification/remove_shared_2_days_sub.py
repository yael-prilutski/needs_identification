import numpy as np
from numpy.linalg import svd
from os.path import join

from configurations import *
from mouse import Mouse
from Processor import Processor
from axes.axes_analyzer import AxesAnalyzer

WEEK = 'week1'
RELEVANT_DAYS = f'{WEEK}_days'
RELEVANT_MICE = ALL_MICE


class clean_water_food_axes(Processor):

    def __init__(self, mouse, vector_name, *args, **kwargs):
        Processor.__init__(self, *args, **kwargs)
        self.mouse = mouse
        self.days = mouse.days
        self.vector_name = vector_name
        self.sec_hz = mouse.sec_hz
        self.week = mouse.week
        self.analyzer = AxesAnalyzer(mouse.sec_hz)

    def run(self):
        vector_location = join(self.week.data_dir_path, 'axes', self.vector_name)
        vectors_dict = np.load(vector_location, allow_pickle=True)[()]

        all_vectors_dict = [vectors_dict[self.days[1].name]['sub'], vectors_dict[self.days[3].name]['sub']]

        M = np.column_stack(all_vectors_dict)
        U, S, Vt = svd(M)
        print(f'S: {S}')
        u1 = U[:, 0]

        alphas = u1 @ M
        print(f'alphas: {alphas}')
        shared = np.outer(u1, alphas)
        M_clean = M - shared
        day_water_clean = M_clean[:, 0]
        day_food_clean = M_clean[:, 1]

        vectors_dict['cleaned_water'] = {
            'sub': day_water_clean,
            'start': np.dot(vectors_dict[self.days[1].name]['start_population'], day_water_clean),
            'end': np.dot(vectors_dict[self.days[1].name]['end_population'], day_water_clean)}
        vectors_dict['cleaned_food'] = {
            'sub': day_food_clean,
            'start': np.dot(vectors_dict[self.days[3].name]['start_population'], day_food_clean),
            'end': np.dot(vectors_dict[self.days[3].name]['end_population'], day_food_clean)}
        np.save(join(self.week.data_dir_path, 'axes', 'clean_axes_2_days.npy'), vectors_dict)


def main():
    for mouse_path in RELEVANT_MICE:
        if WEEK in mouse_path.keys():
            mouse = Mouse(mouse_path, mouse_path[RELEVANT_DAYS], mouse_path[WEEK])
            process = clean_water_food_axes(mouse, vector_name='axes_vector_neutral.npy')
            process.run()
            print(f'finished processing {mouse.name}')


'__main__' == __name__ and main()
