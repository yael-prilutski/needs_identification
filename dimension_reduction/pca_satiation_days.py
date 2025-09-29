import numpy as np
from os.path import join, isdir
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from sklearn.decomposition import PCA

from configurations import *
from mouse import Mouse
from decoder.decoder_analyzer import DecoderAnalyzer

MICE = ALL_MICE
PATH = join(RESULTS_PATH, 'PCA', 'satiation')


class DecoderSatiation:

    def __init__(self, mouse, reward):
        self.mouse = mouse
        self.days = [d for d in mouse.days if d.name[-1] != 'b']
        self.sec_hz = mouse.sec_hz
        self.reward = reward
        self.week_data = mouse.week.data_dir_path
        self.original_iti = self.sec_hz * 8
        self.bin = self.sec_hz
        self.iti = self.original_iti // self.bin
        self.decoder_analyzer = DecoderAnalyzer(self.sec_hz)

    def analyze_day(self, iti_dict):
        train_day = self.decoder_analyzer.bin_data(iti_dict[self.days[0].name]['cells'], self.bin).T
        test_day = self.decoder_analyzer.bin_data(iti_dict[self.days[1].name]['cells'], self.bin).T

        pca = PCA(n_components=10)
        pca.fit(train_day)

        test_pca = pca.transform(test_day).T
        trained_pca = pca.transform(train_day).T

        return test_pca, trained_pca

    def visualize(self, day_data, rewards):
        test, trained = day_data
        if self.reward == 'water':
            color = 'darkblue'
        else:
            color = 'darkred'

        min_size = min(test.shape[1], trained.shape[1])

        plt.figure(figsize=(20, 15))
        plt.suptitle(f'{self.mouse.name} PCA {self.reward} days')
        for i in range(9):
            test_smooth = uniform_filter1d(test[i], size=self.iti * 5)
            trained_smooth = uniform_filter1d(trained[i], size=self.iti * 5)
            correlation = np.corrcoef(test_smooth[:min_size], trained_smooth[:min_size])[0, 1]
            corr_reward = np.corrcoef(trained_smooth, rewards)[0, 1]
            plt.subplot(3, 3, i + 1)
            plt.title(f'Component: {i}, corr reward: {round(corr_reward, 2)}, corr test: {round(correlation, 2)}',
                      fontsize=15)
            plt.plot(test_smooth, color=color, label='test', alpha=0.7)
            plt.plot(trained_smooth, color='gray', label='trained', alpha=0.7)
            plt.legend()

        plt.savefig(join(PATH, f'{self.mouse.name}_pca_{self.reward}.jpg'))

    def cumulative_rewards(self, iti_dict):
        if self.reward == 'water':
            reward = [self.days[0].drank]
        else:
            reward = [self.days[0].ate, self.days[0].omission_food_taste]
        trial_types = iti_dict[self.days[0].name]['trials_type']
        neutral_trials = iti_dict[self.days[0].name]['relevant_trials']

        reward_array = [1 if trial_types[i] in reward else 0 for i in range(len(trial_types))]
        cumulative = np.cumsum(reward_array)
        neutral_trials_rewards = np.hstack([[cumulative[i]] * self.iti for i in neutral_trials])
        return neutral_trials_rewards

    def run(self):
        iti_dict = np.load(join(self.week_data, 'normalized_itis.npy'), allow_pickle=True)[()]
        pca_data = self.analyze_day(iti_dict)
        rewards = self.cumulative_rewards(iti_dict)
        self.visualize(pca_data, rewards)


def main():
    for reward in ['water', 'food']:
        days = f'opto_{reward}_days'
        week = f'opto_{reward}_week'
        for mouse_dict in [MOUSE_YP82]:
            if days in mouse_dict.keys():
                mouse = Mouse(mouse_dict, mouse_dict[days], mouse_dict[week])
                process = DecoderSatiation(mouse, reward)
                process.run()
                print(f'finished mouse {mouse.name}')


'__main__' == __name__ and main()
