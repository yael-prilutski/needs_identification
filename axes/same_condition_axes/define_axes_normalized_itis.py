import numpy as np
from os.path import join

from configurations import *
from mouse import Mouse
from Processor import Processor
from axes.axes_analyzer import AxesAnalyzer
from main_analyzer import MainAnalyzer

RELEVANT_MICE = ALL_MICE


class WithinDayAxis(Processor):

    def __init__(self, mouse, vector_name, *args, **kwargs):
        Processor.__init__(self, *args, **kwargs)
        self.days = [d for d in mouse.days if d.name[-1] != 'b']
        self.vector_name = vector_name
        self.sec_hz = mouse.sec_hz
        self.week = mouse.week
        self.analyzer = AxesAnalyzer(mouse.sec_hz)
        self.main_analyzer = MainAnalyzer(self.sec_hz)

    def find_axes(self, day):
        day_data = day.load_data_dict()
        day_cues = day_data['cues']
        day_trials = day_data['trials_classification'][:-1]
        if day.is_light:
            day_cues = self.main_analyzer.change_to_regular_opto_cues(day, day_cues)
            day_trials = self.main_analyzer.change_to_regular_opto_trials(day, day_trials)

        relevant_trials_index = self.find_relevant_neutral_index(
            day, day_trials, day_data['water_licks'], day_data['pellet_licks'])
        start_index, end_index = self.find_trials_for_axes(day, day_cues, day_trials, relevant_trials_index)

        fix_cells_baseline = [[np.hstack([cell[trial][self.sec_hz * 2:], cell[trial + 1][:self.sec_hz * 2]])
                               for trial in range(len(cell) - 1)] for cell in day_data['cells']]
        day_cells_data = self.main_analyzer.min_max_normalization(fix_cells_baseline)
        start_iti = self.cut_iti(day_cells_data, start_index, do_mean=True)
        end_iti = self.cut_iti(day_cells_data, end_index, do_mean=True)

        return start_iti, end_iti, end_index[-1]

    def vector_data(self, day):
        start_of_session, end_of_session, last_index = self.find_axes(day)

        sub_start_finish = end_of_session - start_of_session
        start_vector = np.dot(start_of_session, sub_start_finish)
        end_vector = np.dot(end_of_session, sub_start_finish)

        return sub_start_finish, start_vector, end_vector, start_of_session, end_of_session, last_index

    def run(self):
        vectors_location = join(self.week.data_dir_path, 'axes', self.vector_name)
        water_food_vectors_dict = {}
        for day in self.days:
            sub, start, end, start_population, end_population, last_index_vector = self.vector_data(day)
            water_food_vectors_dict[day.name] = {
                'sub': sub, 'start': start, 'end': end, 'start_population': start_population,
                'end_population': end_population, 'last_index_vector': last_index_vector}

        np.save(vectors_location, water_food_vectors_dict)


def main():
    for reward in ['water', 'food']:
        days = f'opto_{reward}_days'
        week = f'opto_{reward}_week'
        for mouse_path in RELEVANT_MICE:
            if week in mouse_path.keys():
                mouse = Mouse(mouse_path, mouse_path[days], mouse_path[week])
                process = WithinDayAxis(mouse, vector_name='axes_vector_normalized.npy')
                process.run()
                print(f'finished processing {mouse.name}')


'__main__' == __name__ and main()
