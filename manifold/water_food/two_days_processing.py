import shutil
import numpy as np
from os import remove
from multiprocessing import Pool

from configurations import *
from mouse import Mouse
from manifold import internal
from manifold.visualize_datasets import visualize_dataset, COLOR_MAP, get_cluster_labels

MICE = ALL_MICE
WEEK = 'week1'
DAYS = f'{WEEK}_days'
BASE_PATH = r'E:\code\main_analysis\dimensions_clustering'


class Clustering:

    def __init__(self, mouse):
        self.base_path = BASE_PATH
        self.data_path = join(BASE_PATH, 'processing')
        self.mouse = mouse
        self.week = mouse.week
        self.days = self.mouse.days
        self.sec_hz = mouse.days[0].smooth_factor
        self.bin_value = int(self.sec_hz / 6)

    def single_cell_min_max(self, cell):
        perc_cell_min = np.percentile(cell, 5)
        perc_cell_max = np.percentile(cell, 95)
        normalized_cell = (cell - perc_cell_min) / (perc_cell_max - perc_cell_min)
        return normalized_cell

    def prepare_environment(self, data_name, day1_data, day2_data):
        internal.initialize_environment(join(self.base_path))

        days_matrix = []
        for day in [day1_data, day2_data]:
            concatenated_cells = []
            cells, relevant_trials = day
            for cell in cells:
                relevant_activity = [cell[trial][-self.sec_hz * 6:] for trial in relevant_trials]
                concatenated_cells.append(np.concatenate(relevant_activity, axis=0))

            binned_activity = self.bin_matrix_manually(np.array(concatenated_cells))
            with Pool() as pool:
                normalized_cells = pool.map(self.single_cell_min_max, binned_activity)
                days_matrix.append(normalized_cells)

        two_days_cells = [np.concatenate((days_matrix[0][cell], days_matrix[1][cell]))
                          for cell in range(len(days_matrix[0]))]

        internal.create_dataset(np.array(two_days_cells), data_name)

    def bin_matrix_manually(self, matrix):
        chunks = np.arange(0, ((matrix.shape[1] - matrix.shape[1] % self.bin_value) / self.bin_value))
        list_of_indices_to_bin = [np.arange((int(chunk_idx) * self.bin_value), ((int(chunk_idx) + 1) * self.bin_value))
                                  for chunk_idx in chunks]
        binned_matrix = np.transpose(
            np.vstack([np.mean(matrix[:, indices_to_bin], axis=1) for indices_to_bin in list_of_indices_to_bin]))
        return binned_matrix

    def process_data(self, data_name):
        lem_iter_1_params = {'method': "lem", 'ndim': 20, 'knn': .075} #0.075/ 0.05/ 0.09
        lem_iter_2_params = {'method': "lem", 'ndim': 5, 'knn': .025} #0.025/ 0.015/ 0.04
        dataset_processing = [{'opcode': internal.REDUCE, 'params': lem_iter_1_params, 'alias': "lem_1"},
                              {'opcode': internal.REDUCE, 'params': lem_iter_2_params, 'alias': "lem_final"}]
        internal.process_dataset(dataset_name=data_name, op_list=dataset_processing)
        # lem_1 = internal.get_dataset(data_name, alias="lem_1")
        # est_dim = internal.estimate_dimensionality(matrix=lem_1)
        # print(f"Estimated dimensionality for {data_name} is {est_dim}")

    def cluster_labels(self, data_name):
        cluster_map = internal.process_cluster_map(dataset_name=data_name, alias="lem_final", max_cluster_number=15)

    def run(self):
        internal.initialize_environment(join(self.base_path))
        iti_dict = np.load(
            join(self.week.data_dir_path, 'axes', f'{self.mouse.name}_dot_products_dict.npy'), allow_pickle=True)[()]

        for days in [[0, 1, 'water'], [2, 3, 'food']]:
            day1_i, day2_i, condition = days
            name = f'{WEEK}_{condition}'
            day1 = self.days[day1_i]
            day2 = self.days[day2_i]

            data_name = f'{self.mouse.name}_{name}'
            path_day_data1 = join(self.data_path, 'day_data1.npy')
            path_day_data2 = join(self.data_path, 'day_data2.npy')
            shutil.copy(day1.combined_data_dict_path(), path_day_data1)
            shutil.copy(day2.combined_data_dict_path(), path_day_data2)
            day_data1 = np.load(path_day_data1, allow_pickle=True)[()]
            day_data2 = np.load(path_day_data2, allow_pickle=True)[()]

            self.prepare_environment(data_name, [day_data1['cells'], iti_dict[day1.name]['relevant_trials']],
                                     [day_data2['cells'], iti_dict[day2.name]['relevant_trials']])
            self.process_data(data_name)
            self.cluster_labels(data_name)

            shutil.move(join(BASE_PATH, 'data', data_name), join(MANIFOLD_PATH, name, data_name))
            remove(path_day_data1)
            remove(path_day_data2)

            cluster_labels = get_cluster_labels(data_name, join(MANIFOLD_PATH, name))
            reduced_dataset = np.load(join(MANIFOLD_PATH, name, data_name, 'lem_final.npy'))
            visualize_dataset(reduced_dataset,
                              join(RESULTS_PATH, 'manifold', 'structure', name, f'{self.mouse.name}.jpg'),
                              default=True,
                              variable=cluster_labels,
                              color_map=COLOR_MAP)


def main():
    for mouse_dict in MICE:
        if DAYS in mouse_dict.keys():
            mouse = Mouse(mouse_dict, mouse_dict[DAYS], mouse_dict[WEEK])
            process = Clustering(mouse)
            process.run()


'__main__' == __name__ and main()
