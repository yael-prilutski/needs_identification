import numpy as np
from os.path import join

from configurations import *
from mouse import Mouse
from Processor import Processor
from axes.axes_analyzer import AxesAnalyzer

RELEVANT_DAYS = 'week2_days'
WEEK = 'week2'
RELEVANT_MICE = ALL_MICE


class WithinDayAxis(Processor):

    def __init__(self, mouse, vector_name, *args, **kwargs):
        Processor.__init__(self, *args, **kwargs)
        self.days = mouse.days
        self.vector_name = vector_name
        self.sec_hz = mouse.smooth_factor
        self.week = mouse.week
        self.analyzer = AxesAnalyzer(mouse.smooth_factor)

    def run(self):
        vectors_location = join(self.week.data_dir_path, 'axes', self.vector_name)
        axis_days = [self.days[1], self.days[3]]

        water_food_vectors_dict = {}
        for day in axis_days:
            sub, start, end, start_population, end_population, last_index_vector = self.analyzer.find_vector(day)
            water_food_vectors_dict[day.name] = {
                'sub': sub, 'start': start, 'end': end, 'start_population': start_population,
                'end_population': end_population, 'last_index_vector': last_index_vector}

        np.save(vectors_location, water_food_vectors_dict)


def main():
    for mouse_path in RELEVANT_MICE:
        if WEEK in mouse_path.keys():
            mouse = Mouse(mouse_path, mouse_path[RELEVANT_DAYS], mouse_path[WEEK])
            process = WithinDayAxis(mouse, vector_name='axes_vector_neutral.npy')
            process.run()
            print(f'finished processing {mouse.name}')


'__main__' == __name__ and main()
