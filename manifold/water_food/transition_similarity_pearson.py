import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr
plt.rcParams['svg.fonttype'] = 'none'

from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import AnovaRM

from configurations import *
from mouse import Mouse
from align_heatmaps_by_consumption import AlignByConsumption
from clusters_heatmaps import ClustersHeatmaps
from manifold.manifold_analyzer import ManifoldAnalyzer

MICE = [MOUSE_YP79, MOUSE_YP82, MOUSE_YP83, MOUSE_YP84, MOUSE_YP86]
# MICE = [MOUSE_YP82]
DAYS = 'week1_days'
WEEK_TYPE = 'week1'
SAVE_PATH = join(RESULTS_PATH, 'water_food', 'manifold', 'cluster_transition_dynamics', 'pearson_by_cue')


class ClustersDistribution:

    def __init__(self, mouse):
        self.mouse = mouse
        self.days = self.mouse.days
        self.week_data_path = mouse.week.data_dir_path
        self.values_clusters = [-1, 1, 2, 3, 4, 5, 6, 7]
        self.analyzer = ManifoldAnalyzer()

    @staticmethod
    def truncate_colormap(cmap, minval=0.2, maxval=1.0, n=256):
        new_cmap = LinearSegmentedColormap.from_list(
            f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    def compute_transition_matrix(self, df, n_trials):
        value_to_index = {v: i for i, v in enumerate(self.values_clusters)}
        n_states = len(self.values_clusters)
        transition_counts = np.zeros((n_states, n_states))

        max_length = min([len(df.iloc[row].dropna()) for row in range(min(n_trials, df.shape[0]))])
        filtered_df = df.iloc[:n_trials, :max_length]

        for row in range(filtered_df.shape[0]):
            sequence = filtered_df.iloc[row].tolist()
            for i in range(len(sequence) - 1):
                from_state = value_to_index[sequence[i]]
                to_state = value_to_index[sequence[i + 1]]
                transition_counts[from_state, to_state] += 1

        original_counts = transition_counts.copy()
        threshold = 3
        transition_counts[transition_counts < threshold] = 0
        np.fill_diagonal(transition_counts, 0)
        for i in range(n_states):
            row_max = np.sum(transition_counts[i])
            if row_max > 0:
                transition_counts[i] /= row_max

        return transition_counts

    def visualize_transition_matrix(self, dfs, similarities):
        title = ['drank1', 'drank2', 'ate1', 'ate2', 'both_w', 'both_f', 'day1_w', 'day1_f']
        gray_r_trunc = self.truncate_colormap(plt.cm.gray_r, 0.2, 1.0)

        plt.figure(figsize=(30, 21))
        plt.suptitle(
            f'{self.mouse.name} clusters transition dynamic\nwater: {similarities[0]}, '
            f'food: {similarities[1]}, both: {similarities[2]}, day1: {similarities[3]}', size=30)
        for i in range(len(dfs)):
            ax = plt.subplot(2, 4, i + 1)
            plt.title(title[i], fontsize=25)
            im = ax.imshow(dfs[i], cmap=gray_r_trunc, vmax=1, aspect='equal', interpolation='nearest',
                           rasterized=True)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.xticks(np.arange(len(self.values_clusters)), np.array(self.values_clusters), fontsize=20)
            plt.yticks(np.arange(len(self.values_clusters)), np.array(self.values_clusters), fontsize=20)
            plt.ylabel('From', fontsize=25)
            plt.xlabel('To', fontsize=25)
        plt.savefig(join(SAVE_PATH, f'{self.mouse.name}.svg'))
        plt.close()

    def pearson_test(self, df1, df2):
        threshold = 0.01
        df1[df1 < threshold] = 0
        df1[df2 < threshold] = 0
        mask = (df1 + df2) > 0
        filtered_1 = df1[mask]
        filtered_2 = df2[mask]

        corr, p_value = pearsonr(filtered_1, filtered_2)

        return round(corr, 3)

    def run(self):
        process = ClustersHeatmaps(self.mouse, do_visualize=False)

        d1, d2, a1, a2, bw, bf = process.run()
        d1_w, d1_f = self.analyzer.load_day1_trials(self.week_data_path)

        n_trials = np.hstack([[min(len(p1), len(p2))] * 2 for p1, p2 in [[d1, d2], [a1, a2], [bw, bf], [d1_w, d1_f]]])

        self.compute_transition_matrix(d1_w, n_trials[6])

        drank1, drank2, ate1, ate2, both_w, both_f, day1_w, day1_f = [
            self.compute_transition_matrix(m, n_trials[i])
            for i, m in enumerate([d1, d2, a1, a2, bw, bf, d1_w, d1_f])]

        water_jaccard = self.pearson_test(drank1, drank2)
        food_jaccard = self.pearson_test(ate1, ate2)
        both_jaccard = self.pearson_test(both_w, both_f)
        day1_jaccard = self.pearson_test(day1_w, day1_f)

        self.visualize_transition_matrix(
            [drank1, drank2, ate1, ate2, both_w, both_f, day1_w, day1_f],
            [water_jaccard, food_jaccard, both_jaccard, day1_jaccard])
        return water_jaccard, food_jaccard, both_jaccard, day1_jaccard


def all_mice_visualize(all_mice):
    mice_by_condition = [[m[c] for m in all_mice] for c in range(len(all_mice[0]))]
    mice_names = ['1', '2', '3', '4', '5']
    mice_df = pd.DataFrame({
        'subject': mice_names,
        'Water': mice_by_condition[0],
        'Food': mice_by_condition[1],
        'Both': mice_by_condition[2],
        'Day1': mice_by_condition[3]})
    data_long = pd.melt(mice_df, id_vars=['subject'], value_vars=['Water', 'Food', 'Both', 'Day1'],
                        var_name='Day', value_name='Score')
    rm_anova = AnovaRM(data_long, depvar='Score', subject='subject', within=['Day'])
    anova_results = rm_anova.fit()
    p_value = round(anova_results.anova_table['Pr > F']['Day'], 3)
    tukey = pairwise_tukeyhsd(endog=data_long['Score'], groups=data_long['Day'], alpha=0.05)
    tukey_p = [round(x, 3) for x in tukey.pvalues]

    plt.figure(figsize=(24, 24))
    plt.suptitle(f'All mice pearson similarity \n W-F: {tukey_p[5]}, B-W: {tukey_p[1]}, B-F: {tukey_p[2]}\n '
                 f'Day1-B: {tukey_p[0]}, Day1-W: {tukey_p[4]}, Day1-F: {tukey_p[3]}', size=30)

    conditions = ['water-water', 'food-food', 'water-food', 'water-food day1']
    colors = ['blue', 'red', 'purple', 'mediumorchid']

    for mouse in all_mice:
        plt.plot(conditions, mouse, c='lightgrey', alpha=0.5)
    mean_values = [np.mean(m) for m in mice_by_condition]
    plt.bar(conditions, mean_values, color=colors, alpha=0.5)
    plt.ylabel('Similarity', fontsize=25)
    plt.ylim([0, 1])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # plt.show()
    plt.savefig(join(SAVE_PATH, 'barplot_pearson_similarity.svg'))


def main():
    all_mice_jaccard = []
    for mouse_dict in MICE:
        if DAYS in mouse_dict.keys():
            mouse = Mouse(mouse_dict, mouse_dict[DAYS], mouse_dict[WEEK_TYPE])
            process = ClustersDistribution(mouse)
            all_mice_jaccard.append(process.run())
            print(f'finished mouse {mouse.name}')
    all_mice_visualize(all_mice_jaccard)


'__main__' == __name__ and main()
