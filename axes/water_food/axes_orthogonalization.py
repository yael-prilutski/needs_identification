import numpy as np
from os.path import join

from configurations import *
from mouse import Mouse
from Processor import Processor

RELEVANT_MICE = ALL_MICE
RELEVANT_DAYS = 'week2_days'
WEEK = 'week2'


class Orthogonalization(Processor):

    def __init__(self, mouse, vector_name, *args, **kwargs):
        Processor.__init__(self, *args, **kwargs)
        self.mouse = mouse
        self.week = mouse.week
        self.vector_name = vector_name

    def run(self):
        vectors_dict = np.load(join(self.week.data_dir_path, 'axes', self.vector_name), allow_pickle=True)[()]

        water_vector = vectors_dict[self.mouse.days[1].name]['sub']
        food_vector = vectors_dict[self.mouse.days[3].name]['sub']

        for ortho_axis, comparison_axis, vector_values, new_name in [
                [water_vector, food_vector, vectors_dict[self.mouse.days[1].name], 'water_ortho'],
                [food_vector, water_vector, vectors_dict[self.mouse.days[3].name], 'food_ortho']]:

            proj = (np.dot(ortho_axis, comparison_axis) / np.dot(comparison_axis, comparison_axis)) * comparison_axis
            orthogonalized = ortho_axis - proj
            original_magnitude = np.linalg.norm(ortho_axis)
            orthogonalized_magnitude = np.linalg.norm(orthogonalized)
            normalized = orthogonalized * (original_magnitude / orthogonalized_magnitude)

            start = np.dot(vector_values['start_population'], normalized)
            end = np.dot(vector_values['end_population'], normalized)

            vectors_dict[new_name] = {
                'sub': normalized,
                'start': start,
                'end': end
            }

        np.save(join(self.week.data_dir_path, 'axes', self.vector_name), vectors_dict)


def main():
    for mouse_p in RELEVANT_MICE:
        if WEEK in mouse_p.keys():
            mouse = Mouse(mouse_p, mouse_p[RELEVANT_DAYS], mouse_p[WEEK])
            vector_dict_name = 'axes_vector_neutral.npy'
            process = Orthogonalization(mouse, vector_name=vector_dict_name)
            process.run()
            print(f'finished processing {mouse.name}')


'__main__' == __name__ and main()
