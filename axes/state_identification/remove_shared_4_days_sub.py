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
        iti_dict = np.load(
            join(self.week.data_dir_path, 'axes', f'{self.mouse.name}_iti_dict.npy'), allow_pickle=True)[()]

        all_vectors_dict = {
            self.days[1].name: vectors_dict[self.days[1].name]['sub'],
            self.days[3].name: vectors_dict[self.days[3].name]['sub'],
        }
        for i_day in [0, 2]:
            day = self.days[i_day]
            day_data = day.load_data_dict()
            day_cues = day_data['cues']
            day_trials = day_data['trials_classification'][:-1]
            neutral_trials = iti_dict[day.name]['relevant_trials']
            iti_df = iti_dict[day.name]['iti_df']
            start_index, end_index = self.analyzer.find_trials_for_axes(day, day_cues, day_trials, neutral_trials)

            start_itis = iti_df.iloc[:, :len(start_index) * self.analyzer.iti_slice].mean(axis=1)
            end_itis = iti_df.iloc[:, -len(end_index) * self.analyzer.iti_slice:].mean(axis=1)
            sub_start_finish = end_itis - start_itis
            all_vectors_dict[day.name] = sub_start_finish

        M = np.column_stack([all_vectors_dict[day.name] for day in self.days[:4]])
        U, S, Vt = svd(M)

        day_water_clean = all_vectors_dict[self.days[1].name]
        day_food_clean = all_vectors_dict[self.days[3].name]
        found_component = False
        for i in range(4):
            u = U[:, i]
            alphas = u @ M
            if (np.all(alphas > 0) or np.all(alphas < 0)) and S[i] > 0.5:
                print(f'alphas: {alphas}')
                shared = np.outer(u, alphas)
                M_clean = M - shared
                day_water_clean = M_clean[:, 1]
                day_food_clean = M_clean[:, 3]
                found_component = True
                print(f'Found shared component with singular value {S[i]}')
                break

        vectors_dict['cleaned']: found_component

        if found_component:
            vectors_dict['cleaned_water'] = {
                'sub': day_water_clean,
                'start': np.dot(vectors_dict[self.days[1].name]['start_population'], day_water_clean),
                'end': np.dot(vectors_dict[self.days[1].name]['end_population'], day_water_clean)}
            vectors_dict['cleaned_food'] = {
                'sub': day_food_clean,
                'start': np.dot(vectors_dict[self.days[3].name]['start_population'], day_food_clean),
                'end': np.dot(vectors_dict[self.days[3].name]['end_population'], day_food_clean)}
            np.save(join(self.week.data_dir_path, 'axes', 'clean_axes.npy'), vectors_dict)
        else:
            vectors_dict['cleaned_water'] = vectors_dict[self.days[1].name]
            vectors_dict['cleaned_food'] = vectors_dict[self.days[3].name]


def main():
    for mouse_path in RELEVANT_MICE:
        if WEEK in mouse_path.keys():
            mouse = Mouse(mouse_path, mouse_path[RELEVANT_DAYS], mouse_path[WEEK])
            process = clean_water_food_axes(mouse, vector_name='axes_vector_neutral.npy')
            process.run()
            print(f'finished processing {mouse.name}')
            print('--------------------------------')


'__main__' == __name__ and main()
