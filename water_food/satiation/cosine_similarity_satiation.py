import numpy as np
import pandas as pd
from os.path import join
from scipy.ndimage import uniform_filter1d
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams['svg.fonttype'] = 'none'

from configurations import *
from mouse import Mouse

ANALYSIS_TYPE = 'reward'
PATH = join(RESULTS_PATH, 'water_food', 'satiation', 'satiation_cosine_similarity', ANALYSIS_TYPE)


class PopulationVectorRewards:

    def __init__(self, mouse, reward):
        self.mouse = mouse
        self.days = self.mouse.days
        self.week_data = mouse.week.data_dir_path
        self.reward = reward
        self.smoothing_window = 2
        self.n_bl = 15

    def calculate_population_vector(self, responses):
        responses_smoothed = pd.DataFrame(
            [uniform_filter1d(responses.iloc[cell], size=self.smoothing_window) for cell in range(len(responses))])
        bl = responses_smoothed.iloc[:, :self.n_bl].mean(axis=1)
        results = [
            cosine_similarity(bl.to_numpy().reshape(1, -1), responses_smoothed[trial].to_numpy().reshape(1, -1))[0][0]
            for trial in range(len(responses_smoothed.iloc[0]))]
        return results

    def visualize(self, response):
        plt.figure(figsize=[21, 21])
        plt.suptitle(f'{self.mouse.name} {self.reward} cosine similarity', fontsize=24)
        plt.scatter(range(len(response)), response, s=40, alpha=0.7)
        plt.plot(uniform_filter1d(response, size=5), color='gray', alpha=0.9)
        plt.xlabel('Trials', fontsize=20)
        plt.ylabel(f'Cosine similarity', fontsize=20)
        plt.ylim(min(min(response) - 0.1, 0), max(max(response) + 0.1, 1))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.subplots_adjust(wspace=0.5)
        # plt.show()
        name_fig = f'{self.mouse.name}_{self.reward}_pv.svg'
        plt.savefig(join(PATH, name_fig))
        plt.close()

    def run(self):
        dict_responses = np.load(
            join(self.week_data, 'dict_mean_responses_per_trial.npy'), allow_pickle=True)[()]
        responses = self.calculate_population_vector(pd.DataFrame(dict_responses[f'day1_{ANALYSIS_TYPE}']))
        self.visualize(responses)
        print(f'finished data processing {self.mouse.name}')
        return responses


def mean_mice_pv(all_pv, reward):
    plt.figure(figsize=[21, 21])
    plt.suptitle(f'All mice cosine similarity - {reward}', fontsize=24)

    all_mice_values = [uniform_filter1d(m, 5) for m in all_pv]
    min_value = min(min([min(v) for v in all_mice_values]) - 0.1, 0)
    max_value = max(max([max(v) for v in all_mice_values]) + 0.1, 1)
    max_length = int(np.percentile([len(v) for v in all_mice_values], 80))
    mice_mean = pd.DataFrame(all_mice_values).mean(axis=0)[:max_length].values
    plt.plot(mice_mean, color='darkblue')
    for m in all_mice_values:
        plt.plot(m[:max_length], color='gray', alpha=0.5)
    plt.xlabel('Trials', fontsize=20)
    plt.ylabel('Cosine similarity', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(min_value, max_value)

    plt.savefig(join(PATH, f'all_mice_cosine_similarity_{reward}.jpg'))
    plt.close()


def main():
    for reward in ['water']:
        week_type = f'opto_{reward}_week'
        days = f'opto_{reward}_days'
        all_pv = []
        for mouse_dict in [MOUSE_YP86, MOUSE_YP84]:
            if week_type in mouse_dict.keys():
                mouse = Mouse(mouse_dict, mouse_dict[days], mouse_dict[week_type])
                processor = PopulationVectorRewards(mouse, reward)
                all_pv.append(processor.run())
        # mean_mice_pv(all_pv, reward)


'__main__' == __name__ and main()
