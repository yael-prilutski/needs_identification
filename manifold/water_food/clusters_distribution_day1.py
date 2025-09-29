import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from configurations import *
from mouse import Mouse
from manifold.manifold_analyzer import ManifoldAnalyzer

MICE = [MOUSE_YP79, MOUSE_YP82, MOUSE_YP83, MOUSE_YP84, MOUSE_YP86, MOUSE_YP64]
DAYS = 'week1_days'
WEEK_TYPE = 'week1'
REWARD_ANALYSIS = False
SAVE_PATH = join(RESULTS_PATH, 'water_food', 'manifold', 'day1_distribution_similarity')
if REWARD_ANALYSIS:
    SAVE_PATH = join(SAVE_PATH, 'by_reward')
else:
    SAVE_PATH = join(SAVE_PATH, 'by_cue')


class ClustersDistributionDay1:

    def __init__(self, mouse):
        self.mouse = mouse
        # self.reward_analysis = reward_analysis
        self.week_data = mouse.week.data_dir_path
        self.sec_hz = mouse.days[0].smooth_factor
        self.analyzer = ManifoldAnalyzer()

    def calculate_similarity(self, day1, day2):
        day1_distribution, day2_distribution = self.analyzer.calculate_clusters_distribution(day1, day2)
        resized_day1, resized_day2 = self.analyzer.adjust_different_size_arrays(day1_distribution, day2_distribution)

        all_clusters_similarity = []
        i_f = 1
        plt.figure(figsize=(24, 24))
        for cluster in range(len(resized_day1)):
            if max(resized_day1[cluster]) > 0.2 or max(resized_day2[cluster]) > 0.2:
                if max(resized_day1[cluster]) < 0.1:
                    resized_day1[cluster] = [0] * len(resized_day1[cluster])
                if max(resized_day2[cluster]) < 0.1:
                    resized_day2[cluster] = [0] * len(resized_day2[cluster])
                similarity = cosine_similarity(
                    np.array(resized_day1[cluster]).reshape(1, -1),
                    np.array(resized_day2[cluster]).reshape(1, -1))[0][0]

                plt.subplot(4, 2, i_f)
                plt.plot(resized_day1[cluster])
                plt.plot(resized_day2[cluster])
                plt.title(f'{cluster + 1}, {similarity}', fontsize=25)
                i_f += 1
                all_clusters_similarity.append(similarity)

        plt.suptitle(
            f'{self.mouse.name} day1 clusters similarity: {np.mean(all_clusters_similarity)}', fontsize=30)
        plt.savefig(join(SAVE_PATH, f'{self.mouse.name}_day1_clusters_similarity.jpg'))
        plt.close()
        return np.mean(all_clusters_similarity)

    def load_trials_type(self, clusters):
        dict_responses = np.load(
            join(self.mouse.week.data_dir_path, 'rewards_indexes_onsets.npy'), allow_pickle=True)[()]['day1']
        drank_trials = pd.DataFrame([clusters[i] for i in dict_responses['index_drank']])
        ate_trials = pd.DataFrame([clusters[i] for i in dict_responses['index_ate']])
        return drank_trials, ate_trials

    def run(self):
        trials_clusters = np.load(join(self.week_data, 'clusters_by_trials_day1.npy'), allow_pickle=True)[()]['clusters']
        trials_drank, trials_ate = self.load_trials_type(trials_clusters)
        both_similarity = self.calculate_similarity(trials_drank, trials_ate)
        return both_similarity


def all_mice_visualize(all_mice):
    plt.figure(figsize=(12, 12))
    plt.suptitle(f'All mice day1 clusters similarity', size=30)

    plt.bar('similarity', np.mean(all_mice))
    for mouse in all_mice:
        plt.scatter('similarity', mouse, c='black', alpha=0.5)
    plt.ylabel('Percentile', fontsize=25)
    plt.ylim([-0.1, 1])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # plt.show()
    plt.savefig(join(SAVE_PATH, 'barplot_clusters_similarity.jpg'))


def main():
    all_mice_similarities = []
    for mouse_dict in MICE:
        if DAYS in mouse_dict.keys():
            mouse = Mouse(mouse_dict, mouse_dict[DAYS], mouse_dict[WEEK_TYPE])
            process = ClustersDistributionDay1(mouse)
            all_mice_similarities.append(process.run())
            print(f'finished mouse {mouse.name}')
    all_mice_visualize(all_mice_similarities)


'__main__' == __name__ and main()
