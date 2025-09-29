import shutil
import numpy as np
from os import remove

from configurations import *
from mouse import Mouse
from manifold import internal
from manifold.visualize_datasets import visualize_dataset, COLOR_MAP, get_cluster_labels

MICE = [MOUSE3]
WEEK = 'week2'
DAYS = f'{WEEK}_days'
BASE_PATH = r'E:\code\main_analysis\dimensions_clustering'


class Clustering:

    def __init__(self, mouse):
        self.base_path = BASE_PATH
        self.data_path = join(BASE_PATH, 'processing')
        self.mouse = mouse
        self.week = mouse.week
        self.days = self.mouse.days
        self.sec_hz = mouse.days[0].sec_hz
        self.bin_value = int(self.sec_hz / 6)

    def prepare_environment(self, day_cells, data_name, relevant_trials):
        internal.initialize_environment(join(self.base_path))

        concatenated_cells = []
        for cell in day_cells:
            relevant_activity = [cell[trial][-self.sec_hz * 6:] for trial in relevant_trials]
            concatenated_cells.append(np.concatenate(relevant_activity, axis=0))

        binned_activity = self.bin_matrix_manually(np.array(concatenated_cells))
        normalized_cells = []
        for i_cell in range(len(binned_activity)):
            cell = binned_activity[i_cell]
            perc_cell_min = np.percentile(cell, 5)
            perc_cell_max = np.percentile(cell, 95)
            normalized_cells.append((cell - perc_cell_min) / (perc_cell_max - perc_cell_min))

        internal.create_dataset(np.array(normalized_cells), data_name)

    def bin_matrix_manually(self, matrix):
        chunks = np.arange(0, ((matrix.shape[1] - matrix.shape[1] % self.bin_value) / self.bin_value))
        list_of_indices_to_bin = [np.arange((int(chunk_idx) * self.bin_value), ((int(chunk_idx) + 1) * self.bin_value))
                                  for chunk_idx in chunks]
        binned_matrix = np.transpose(
            np.vstack([np.mean(matrix[:, indices_to_bin], axis=1) for indices_to_bin in list_of_indices_to_bin]))
        return binned_matrix

    def process_data(self, data_name):
        lem_iter_1_params = {'method': "lem", 'ndim': 20, 'knn': .075} #0.075/ 0.05/ 0.09
        lem_iter_2_params = {'method': "lem", 'ndim': 4, 'knn': .025} #0.025/ 0.015/ 0.04
        dataset_processing = [{'opcode': internal.REDUCE, 'params': lem_iter_1_params, 'alias': "lem_1"},
                              {'opcode': internal.REDUCE, 'params': lem_iter_2_params, 'alias': "lem_final"}]
        internal.process_dataset(dataset_name=data_name, op_list=dataset_processing)
        lem_1 = internal.get_dataset(data_name, alias="lem_1")
        est_dim = internal.estimate_dimensionality(matrix=lem_1)
        print(f"Estimated dimensionality for {data_name} is {est_dim}")

    def cluster_labels(self, data_name):
        cluster_map = internal.process_cluster_map(dataset_name=data_name, alias="lem_final", max_cluster_number=15)

    def run(self):
        internal.initialize_environment(join(self.base_path))
        iti_dict = np.load(
            join(self.week.data_dir_path, 'axes', f'{self.mouse.name}_dot_products_dict.npy'), allow_pickle=True)[()]

        for day_i in [2]:
            if WEEK == 'week1':
                name = f'day{day_i + 1}'
            else:
                name = f'week2_day{day_i + 1}'
            day = self.days[day_i]

            data_name = f'{self.mouse.name}_{name}'
            path_day_data = join(self.data_path, 'day_data.npy')
            shutil.copy(day.combined_data_dict_path(), path_day_data)
            day_data = np.load(path_day_data, allow_pickle=True)[()]

            self.prepare_environment(day_data['cells'], data_name, iti_dict[day.name]['relevant_trials'])
            self.process_data(data_name)
            self.cluster_labels(data_name)

            shutil.move(join(BASE_PATH, 'data', data_name), join(MANIFOLD_PATH, name, data_name))
            remove(path_day_data)

            cluster_labels = get_cluster_labels(data_name, join(MANIFOLD_PATH, name))
            reduced_dataset = np.load(join(MANIFOLD_PATH, name, data_name, 'lem_final.npy'))
            visualize_dataset(reduced_dataset,
                              join(RESULTS_PATH, 'manifold', 'structure', f'{name}_manifold', f'{self.mouse.name}.jpg'),
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
