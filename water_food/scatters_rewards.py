import numpy as np
import pandas as pd
from os.path import join, isdir
import matplotlib.pyplot as plt
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ttest_rel
plt.rcParams['svg.fonttype'] = 'none'

from configurations import *
from mouse import Mouse
from Visualizer import Visualizer

MICE = ALL_MICE
WEEK_TYPE = f'week1'
DAYS = f'week1_days'
BY_REWARD = True
PATH = join(RESULTS_PATH, 'water_food', 'correlations')
if BY_REWARD:
    PATH = join(PATH, 'by_reward')
else:
    PATH = join(PATH, 'by_cue')


class RewardsCorrelationStability:

    def __init__(self, mouse, w_f_analysis):
        self.mouse = mouse
        self.w_f_analysis = w_f_analysis
        self.days = self.mouse.days
        self.sec_hz = mouse.smooth_factor
        self.dict_responses = join(mouse.week.data_dir_path, 'dict_mean_responses.npy')
        self.visualizer = Visualizer(self.sec_hz)

    def visualize(self, correlation_pairs, correlation_names, labels_names):
        if self.w_f_analysis:
            suptitle = f'{self.mouse.name} rewards - correlations water-food'
            name = f'{self.mouse.name}_correlation_reward_water_food.svg'
        else:
            suptitle = f'{self.mouse.name} rewards - correlations needs'
            name = f'{self.mouse.name}_correlation_reward_needs.svg'

        plt.figure(figsize=(21, 21))
        plt.suptitle(suptitle, size=25)
        for i in range(len(correlation_names)):
            plt.subplot(2, 2, i + 1)
            self.visualizer.scatter_plot_config(correlation_pairs[i], labels_names[i], correlation_names[i])
        # plt.show()
        if not BY_REWARD:
            name = name.replace('reward', 'cue')
        plt.savefig(join(PATH, name))
        plt.close()

    def run(self):
        dict_r = np.load(self.dict_responses, allow_pickle=True)[()]
        if BY_REWARD:
            r_type = 'reward'
        else:
            r_type = 'cue'

        if self.w_f_analysis:
            correlation_names = ['water_1_2', 'food_3_4', 'water_food_1', 'water2_food_4']
            labels_names = [['water_1', 'water_2'], ['food_3', 'food_4'], ['water_1', 'food_1'], ['water_2', 'food_4']]
            correlations_pairs = [
                [dict_r['day_1'][f'water_{r_type}'], dict_r['day_2'][f'water_{r_type}']],
                [dict_r['day_3'][f'food_{r_type}'], dict_r['day_4'][f'food_{r_type}']],
                [dict_r['day_1'][f'water_{r_type}'], dict_r['day_1'][f'food_{r_type}']],
                [dict_r['day_2'][f'water_{r_type}'], dict_r['day_4'][f'food_{r_type}']]]
        else:
            correlation_names = ['water_1_2', 'water_1_3', 'food_3_4', 'food_3_1', 'water_2_food_4']
            labels_names = [['water_1', 'water_2'], ['food_3', 'food_4'], ['water_1', 'water_3'], ['food_3', 'food_1'],
                            ['water_2', 'food_4']]
            correlations_pairs = [
                [dict_r['day_1'][f'water_{r_type}'], dict_r['day_2'][f'water_{r_type}']],
                [dict_r['day_1'][f'water_{r_type}'], dict_r['day_3'][f'water_{r_type}']],
                [dict_r['day_3'][f'food_{r_type}'], dict_r['day_4'][f'food_{r_type}']],
                [dict_r['day_3'][f'food_{r_type}'], dict_r['day_1'][f'food_{r_type}']],
                [dict_r['day_2'][f'water_{r_type}'], dict_r['day_4'][f'food_{r_type}']]]

        filtered_pairs = []
        for pair in correlations_pairs:
            if len(pair[0]) == 0 or len(pair[1]) == 0:
                filtered_pairs.append([[], []])
                continue
            outlines = [i for i in range(len(pair[0])) if abs(pair[0][i]) > 1 or abs(pair[1][i]) > 1]
            filtered_day1 = [pair[0][i] for i in range(len(pair[0])) if i not in outlines]
            filtered_day2 = [pair[1][i] for i in range(len(pair[1])) if i not in outlines]
            filtered_pairs.append([filtered_day1, filtered_day2])

        # self.visualize(filtered_pairs, correlation_names, labels_names)

        all_correlations = [np.corrcoef(pair[0], pair[1])[0, 1] if len(pair[0]) and len(pair[1]) else np.nan
                            for pair in filtered_pairs]

        all_slopes = [np.polyfit(pair[0], pair[1], 1)[0] if len(pair[0]) and len(pair[1]) else np.nan
                      for pair in filtered_pairs]

        print(f'finished figures creation for {self.mouse.name}')
        return all_correlations, all_slopes


def barplot_config(responses, conditions, x_labels, colors, title):
    mice_by_condition = [[m[c] for m in responses] for c in range(len(responses[0]))]
    mice_names = [str(i) for i in range(1, len(responses) + 1)]
    mice_df = pd.DataFrame({
        'subject': mice_names,
        'Water': mice_by_condition[0],
        'Food': mice_by_condition[1],
        conditions[0]: mice_by_condition[2],
        conditions[1]: mice_by_condition[3]})
    data_long = pd.melt(mice_df, id_vars=['subject'], value_vars=['Water', 'Food', conditions[0], conditions[1]],
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

    mice_mean = [np.nanmean(m) for m in mice_by_condition]
    plt.title(title, size=25)
    plt.bar(x_labels, mice_mean, color=colors)
    for m in responses:
        plt.plot(x_labels, m, c='gray', alpha=0.3)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=20)


def anova_needs(responses):
    mice_names = [str(i) for i in range(1, len(responses[0]) + 1)]
    mice_df = pd.DataFrame({
        'subject': mice_names,
        'diff': responses[0],
        'same': responses[1],
        'water_food': responses[2]})
    data_long = pd.melt(mice_df, id_vars=['subject'], value_vars=['diff', 'same', 'water_food'],
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
    return tukey_p


def create_bar_plot_needs(all_correlations, all_slopes):
    conditions = ['Correlations', 'Slopes']
    x_labels = [['water_1_2', 'water_1_3', 'water2_food4'], ['food_3_4', 'food_1_3', 'water2_food4']]
    name = f'summary_correlation_reward_needs.svg'
    colors = [['darkblue', 'blue', 'purple'], ['darkred', 'red', 'purple']]

    plt.figure(figsize=(30, 21))
    plt.suptitle('Rewards responses - correlation summary needs', size=30)

    for i, data in enumerate([all_correlations, all_slopes]):
        same_water = [m[0] for m in data if not np.isnan(m[1])]
        different_water = [m[1] for m in data if not np.isnan(m[1])]
        same_food = [m[2] for m in data if not np.isnan(m[3])]
        different_food = [m[3] for m in data if not np.isnan(m[3])]
        water_food_for_water = [m[4] for m in data if not np.isnan(m[1])]
        water_food_for_food = [m[4] for m in data if not np.isnan(m[3])]
        p_water = anova_needs([different_water, same_water, water_food_for_water])
        p_food = anova_needs([different_food, same_food, water_food_for_food])
        # test_water = round(ttest_rel([same_water[i] for i in range(len(same_water))
        #                               if not np.isnan(different_water[i])],
        #                              [i for i in different_water if not np.isnan(i)])[1], 4)
        # test_food = round(ttest_rel([same_food[i] for i in range(len(same_food))
        #                               if not np.isnan(different_food[i])],
        #                              [i for i in different_food if not np.isnan(i)])[1], 4)

        plt.subplot(2, 2, i + 1)
        plt.title(f'{conditions[i]} water {p_water}', size=25)
        plt.bar(x_labels[0], [np.mean(same_water), np.mean(different_water), np.mean(water_food_for_water)], color=colors[0])
        for mouse in data:
            if not np.isnan(mouse[0]):
                plt.plot(x_labels[0], [mouse[0], mouse[1], mouse[4]], c='gray', alpha=0.3)

        plt.subplot(2, 2, i + 3)
        plt.title(f'{conditions[i]} food {p_food}', size=25)
        plt.bar(x_labels[1], [np.mean(same_food), np.mean(different_food), np.mean(water_food_for_food)], color=colors[1])
        for mouse in data:
            plt.plot(x_labels[1], mouse[2:], c='gray', alpha=0.3)

    if not BY_REWARD:
        name = name.replace('reward', 'cue')

    plt.savefig(join(PATH, name))
    plt.close()


def create_bar_plots(all_correlations, all_slopes):
    conditions = ['Day1', 'Both']
    x_labels = ['water_1_2', 'food_3_4', 'water_food_1', 'water2_food_4']
    suptitle = 'Rewards responses - correlation summary water-food'
    name = f'summary_correlation_reward_water_food.svg'
    colors = ['darkblue', 'darkred', 'orchid', 'purple']

    plt.figure(figsize=(30, 21))
    plt.suptitle(suptitle, size=30)

    plt.subplot(1, 2, 1)
    barplot_config(all_correlations, conditions, x_labels, colors, 'correlations')
    plt.subplot(1, 2, 2)
    barplot_config(all_slopes, conditions, x_labels, colors, 'slopes')

    if not BY_REWARD:
        name = name.replace('reward', 'cue')

    plt.savefig(join(PATH, name))
    plt.close()
    # plt.show()


def main():
    for w_f_analysis in [False]:
        all_correlations = []
        all_slopes = []
        for mouse_dict in MICE:
            if WEEK_TYPE in mouse_dict.keys():
                mouse = Mouse(mouse_dict, mouse_dict[DAYS], mouse_dict[WEEK_TYPE])
                processor = RewardsCorrelationStability(mouse, w_f_analysis)
                corr, slope = processor.run()
                all_correlations.append(corr)
                all_slopes.append(slope)
        if w_f_analysis:
            create_bar_plots(all_correlations, all_slopes)
        else:
            create_bar_plot_needs(all_correlations, all_slopes)


'__main__' == __name__ and main()
