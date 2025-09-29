import numpy as np
import pandas as pd
from os.path import join


class ManifoldAnalyzer:

    def calculate_clusters_distribution(self, day1, day2):
        n_trials_distribution = min(len(day1), len(day2))
        len_day1 = int(np.median([len(day1.iloc[t].dropna()) for t in range(n_trials_distribution)]))
        len_day2 = int(np.median([len(day2.iloc[t].dropna()) for t in range(n_trials_distribution)]))
        day1_distribution = []
        day2_distribution = []
        for cluster in range(1, 8):
            day1_distribution.append([
                (day1.iloc[:n_trials_distribution, i].value_counts().get(cluster, 0)) / n_trials_distribution
                for i in range(len_day1)])
            day2_distribution.append([
                (day2.iloc[:n_trials_distribution, i].value_counts().get(cluster, 0)) / n_trials_distribution
                for i in range(len_day2)])
        return day1_distribution, day2_distribution

    def adjust_different_size_arrays(self, arrays1, arrays2):
        if len(arrays1[0]) == len(arrays2[0]):
            return arrays1, arrays2
        elif len(arrays1[0]) < len(arrays2[0]):
            new_arrays1 = []
            for array in arrays1:
                new_array = np.interp(np.linspace(0, 1, len(arrays2[0])), np.linspace(0, 1, len(array)), array)
                new_arrays1.append(new_array)
            return new_arrays1, arrays2
        else:
            new_arrays2 = []
            for array in arrays2:
                new_array = np.interp(np.linspace(0, 1, len(arrays1[0])), np.linspace(0, 1, len(array)), array)
                new_arrays2.append(new_array)
            return arrays1, new_arrays2

    def load_day1_trials(self, week_data_path):
        clusters = np.load(join(week_data_path, 'manifold', 'clusters_by_trials_day1.npy'), allow_pickle=True)[()]['clusters']
        dict_responses = np.load(join(week_data_path, 'rewards_indexes_onsets.npy'), allow_pickle=True)[()]['day1']
        drank_trials = pd.DataFrame([clusters[i] for i in dict_responses['index_drank']])
        ate_trials = pd.DataFrame([clusters[i] for i in dict_responses['index_ate']])
        return drank_trials, ate_trials

