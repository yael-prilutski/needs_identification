import numpy as np
import pandas as pd
from os.path import join, isdir
import matplotlib.pyplot as plt
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
plt.rcParams['svg.fonttype'] = 'none'

from configurations import *
from mouse import Mouse
from Visualizer import Visualizer
from main_analyzer import MainAnalyzer

MICE = ALL_MICE
PATH = join(RESULTS_PATH, 'water_food', 'satiation', 'correlation_normalized_behavior')


class RewardsCorrelationStability:

    def __init__(self, mouse, reward):
        self.mouse = mouse
        self.reward = reward
        self.days = [d for d in self.mouse.days if d.name[-1] != 'b']
        self.data_dir = mouse.week.data_dir_path
        self.sec_hz = mouse.smooth_factor
        self.dict_onsets = join(self.data_dir, 'rewards_indexes_onsets.npy')
        self.visualizer = Visualizer(self.sec_hz)
        self.main_analyzer = MainAnalyzer(self.sec_hz)

    def visualize(self, correlation_pairs):
        correlation_names = [
            f'{self.reward} original', f'{self.reward} normalized', f'{self.reward} days 1-2 normalized']
        labels_names = [['start', 'end'], ['start', 'end'], ['start1', 'start2']]
        plt.figure(figsize=(28, 15))
        plt.suptitle(f'{self.mouse.name} rewards - correlations licks normalized {self.reward}', size=25)
        for i in range(len(correlation_names)):
            plt.subplot(1, 3, i + 1)
            self.visualizer.scatter_plot_config(
                correlation_pairs[i], labels_names[i], correlation_names[i], include_slope=True)
        # plt.show()
        plt.savefig(join(PATH, f'{self.mouse.name}_correlation_{self.reward}_licks_comparison.jpg'))
        plt.close()

    def individual_licks(self, trial):
        new_trial = np.zeros(len(trial), dtype=int)
        licks = np.where(np.array(trial) == 1)[0]
        if len(licks) == 0:
            return 0
        first_licks = np.array([i for i in licks if i - 1 not in licks])
        new_trial[first_licks] = 1
        return sum(new_trial)

    def compare_licks(self, full_index_start, full_index_end, index_start, index_end, licks_start, licks_end,
                      cells_responses_start, cells_responses_end):
        relevant_trials_start = []
        relevant_trials_end = []
        for i_trial, trial in enumerate(index_end):
            timing = licks_end[i_trial]
            if timing == 0:
                continue
            relevant_start = [i for i, v in enumerate(licks_start) if timing == v]
            relevant_start_index = [index_start[i] for i in relevant_start]
            if len(relevant_start) > 0:
                relevant_trials_start.append(relevant_start_index)
                relevant_trials_end.append(trial)

        responses_start, responses_end = [], []
        if len(relevant_trials_start) > 5 and len(relevant_trials_end) > 5:
            final_index_end = [i for i, v in enumerate(full_index_end) if v in relevant_trials_end]
            responses_end = [np.mean([cell[t] for t in final_index_end]) for cell in cells_responses_end]

            for cell in cells_responses_start:
                cell_start_responses = []
                for trials_group in relevant_trials_start:
                    relevant_index_start = [i for i, v in enumerate(full_index_start) if v in trials_group]
                    cell_start_responses.append(np.mean([cell[t] for t in relevant_index_start]))
                responses_start.append(np.mean(cell_start_responses))

        return responses_start, responses_end

    def analyze_rewards(self, cells_responses):
        day1 = self.days[0]
        day2 = self.days[1]

        responses_day1 = cells_responses['day1_reward']
        responses_day2 = cells_responses['day2_reward']

        if self.reward == 'food':
            onsets_dict = np.load(join(self.data_dir, 'food_consumption_onsets.npy'), allow_pickle=True)[()]
            onsets_dict1 = onsets_dict[day1.name]
            licks1 = day1.load_data_dict()['pellet_licks']
            index1 = onsets_dict1['ate_index']
            window_size = int(len(index1) * 0.33)
            index_start = index1[:window_size]
            latency_start = onsets_dict1['reward_ate'][:window_size]
            index_end = index1[-window_size:]
            latency_end = onsets_dict1['reward_ate'][-window_size:]
            licks_start = [self.individual_licks(licks1[v][latency_start[i] - self.sec_hz: latency_start[i]])
                           for i, v in enumerate(index_start)]
            licks_end = [self.individual_licks(licks1[v][latency_end[i] - self.sec_hz: latency_end[i]])
                         for i, v in enumerate(index_end)]

            onsets_dict2 = onsets_dict[day2.name]
            licks2 = day2.load_data_dict()['pellet_licks']
            index2 = onsets_dict2['ate_index'][:300]
            index_start2 = index2[:window_size]
            onset_start2 = onsets_dict2['reward_ate'][:300][:window_size]
            licks_start2 = [self.individual_licks(
                licks2[v][onset_start2[i] - self.sec_hz: onset_start2[i] + self.sec_hz])
                for i, v in enumerate(index_start2)]

        else:
            onsets_dict = np.load(join(self.data_dir, 'water_licks_onsets.npy'), allow_pickle=True)[()]
            onsets_dict1 = onsets_dict[day1.name]
            licks1 = day1.load_data_dict()['water_licks']
            index1 = onsets_dict1['drank_index'][:300]

            window_size = min(50, int(len(index1) * 0.3))
            # window_size = 30
            index_start = index1[:window_size]
            onset_start = onsets_dict1['reward_drank'][:300][:window_size]
            index_end = index1[-window_size:]
            onset_end = onsets_dict1['reward_drank'][:300][-window_size:]
            licks_start = [self.individual_licks(licks1[v][onset_start[i] - self.sec_hz: onset_start[i] + self.sec_hz])
                           for i, v in enumerate(index_start)]
            licks_end = [self.individual_licks(licks1[v][onset_end[i] - self.sec_hz: onset_end[i] + self.sec_hz])
                         for i, v in enumerate(index_end)]

            onsets_dict2 = onsets_dict[day2.name]
            licks2 = day2.load_data_dict()['water_licks']
            index2 = onsets_dict2['drank_index'][:300]
            index_start2 = index2[:window_size]
            onset_start2 = onsets_dict2['reward_drank'][:300][:window_size]
            licks_start2 = [self.individual_licks(
                licks2[v][onset_start2[i] - self.sec_hz: onset_start2[i] + self.sec_hz])
                for i, v in enumerate(index_start2)]

        responses_start, responses_end = self.compare_licks(
            index1, index1, index_start, index_end, licks_start, licks_end, responses_day1, responses_day1)

        _, responses_start_day2 = self.compare_licks(
            index1, index2, index_start, index_start2, licks_start, licks_start2, responses_day1, responses_day2)

        return responses_start, responses_end, responses_start_day2

    def run(self):
        cells_responses = np.load(
            join(self.data_dir, 'dict_mean_responses_per_trial.npy'), allow_pickle=True)[()]
        original_start = [np.mean(cell[:30]) for cell in cells_responses['day1_reward']]
        original_end = [np.mean(cell[-30:]) for cell in cells_responses['day1_reward']]

        start, end, start2 = self.analyze_rewards(cells_responses)

        correlations_pairs = [[original_start, original_end], [start, end], [start, start2]]

        filtered_pairs = []
        for pair in correlations_pairs:
            if len(pair[0]) == 0 or len(pair[1]) == 0:
                filtered_pairs.append([[], []])
                continue
            outlines = [i for i in range(len(pair[0])) if abs(pair[0][i]) > 1 or abs(pair[1][i]) > 1]
            filtered_day1 = [pair[0][i] for i in range(len(pair[0])) if i not in outlines]
            filtered_day2 = [pair[1][i] for i in range(len(pair[1])) if i not in outlines]
            filtered_pairs.append([filtered_day1, filtered_day2])

        self.visualize(filtered_pairs)

        all_correlations = [np.corrcoef(pair[0], pair[1])[0, 1] if len(pair[0]) and len(pair[1]) else np.nan
                            for pair in filtered_pairs]

        all_slopes = [np.polyfit(pair[0], pair[1], 1)[0] if len(pair[0]) and len(pair[1]) else np.nan
                      for pair in filtered_pairs]

        print(f'finished figures creation for {self.mouse.name}')
        return all_correlations, all_slopes


def anova_calculation(mice_by_condition):
    mice_names = [str(i) for i in range(len(mice_by_condition[0]))]
    mice_df = pd.DataFrame({
        'subject': mice_names,
        'original': mice_by_condition[0],
        'matching': mice_by_condition[1],
        'two_days': mice_by_condition[2]})
    data_long = pd.melt(mice_df, id_vars=['subject'], value_vars=['original', 'matching', 'two_days'],
                        var_name='Condition', value_name='Score')

    complete_subjects = data_long.groupby('subject').filter(lambda x: x['Score'].notna().all())
    rm_anova = AnovaRM(complete_subjects, depvar='Score', subject='subject', within=['Condition'])
    anova_results = rm_anova.fit()
    p_value = round(anova_results.anova_table['Pr > F']['Condition'], 3)
    print(p_value)
    tukey = pairwise_tukeyhsd(endog=complete_subjects['Score'], groups=complete_subjects['Condition'], alpha=0.05)
    tukey_p = [round(x, 3) for x in tukey.pvalues]
    print(tukey)
    print(tukey_p)


def barplot_config(responses, x_labels, colors, title):
    normalized_mice = [m / m[2] for m in responses if not np.isnan(m[1]) and m[2] > 0.3]
    mice_by_condition = [[m[c] for m in normalized_mice] for c in range(len(normalized_mice[0]))]
    ttest = anova_calculation(mice_by_condition)
    mice_mean = [np.nanmean(m) for m in mice_by_condition]
    plt.title(title, size=25)
    plt.bar(x_labels, mice_mean, color=colors)
    for m in normalized_mice:
        plt.plot(x_labels, m, c='gray', alpha=0.3)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=20)


def create_bar_plots(all_correlations, all_slopes, reward):
    x_labels = [f'{reward} original', f'{reward} normalized', f'{reward} days 1-2 normalized']
    colors = ['gray', 'green', 'red']

    plt.figure(figsize=(30, 21))
    plt.suptitle(f'Summary normalized satiaiton {reward}', size=30)

    plt.subplot(1, 2, 1)
    barplot_config(all_correlations, x_labels, colors, 'correlations')
    plt.subplot(1, 2, 2)
    barplot_config(all_slopes, x_labels, colors, 'slopes')

    plt.savefig(join(PATH, f'summary_normalized_satiation_{reward}.svg'))
    plt.close()
    # plt.show()


def main():
    for reward in ['water']:
        week_type = f'opto_{reward}_week'
        days = f'opto_{reward}_days'
        all_correlations = []
        all_slopes = []
        for mouse_dict in ALL_MICE:
            if week_type in mouse_dict.keys():
                mouse = Mouse(mouse_dict, mouse_dict[days], mouse_dict[week_type])
                corr, slope = RewardsCorrelationStability(mouse, reward).run()
                all_correlations.append(corr)
                all_slopes.append(slope)
        create_bar_plots(all_correlations, all_slopes, reward)


'__main__' == __name__ and main()
