import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

MIN_SATIATION_TRIALS = 20


class DecoderAnalyzer:

    def __init__(self, sec_hz):
        self.sec_hz = sec_hz

    def color_after_behavior(self, day_data, day, iti_size):
        relevant_trials = day_data['relevant_trials']
        trials_type = day_data['trials_classification']

        colors = []
        for i in relevant_trials:
            if trials_type[i - 1] == day.neutral or i == 0:
                colors.extend(['gray'] * iti_size)
            elif trials_type[i - 1] in [day.ate, day.omission_food_taste]:
                colors.extend(['red'] * iti_size)
            elif trials_type[i - 1] in [day.not_ate, day.not_ate_licked]:
                colors.extend(['darkred'] * iti_size)
            elif trials_type[i - 1] == day.omission_pellet:
                colors.extend(['brown'] * iti_size)
            elif trials_type[i - 1] == day.drank:
                colors.extend(['lightblue'] * iti_size)
            elif trials_type[i - 1] == day.not_drank:
                colors.extend(['darkblue'] * iti_size)
            elif trials_type[i - 1] == day.omission_water:
                colors.extend(['green'] * iti_size)
            else:
                colors.extend(['black'] * iti_size)
        return colors

    def color_predict_behavior(self, day_data, day, iti_size):
        relevant_trials = day_data['relevant_trials']
        trials_type = day_data['trials_classification']

        colors = []
        for i in relevant_trials:
            if i == relevant_trials[-1]:
                colors.extend(['gray'] * iti_size)
                continue
            trial = trials_type[i + 1]
            if trial == day.neutral:
                colors.extend(['gray'] * iti_size)
            elif trial in [day.ate, day.omission_food_taste]:
                colors.extend(['red'] * iti_size)
            elif trial in [day.not_ate, day.not_ate_licked]:
                colors.extend(['darkred'] * iti_size)
            elif trial == day.omission_pellet:
                colors.extend(['brown'] * iti_size)
            elif trial == day.drank:
                colors.extend(['lightblue'] * iti_size)
            elif trial == day.not_drank:
                colors.extend(['darkblue'] * iti_size)
            elif trial == day.omission_water:
                colors.extend(['green'] * iti_size)
            else:
                colors.extend(['black'] * iti_size)
        return colors

    def blocks_onsets(self, day_data, day, iti_size):
        relevant_trials = day_data['relevant_trials']
        trials_type = day_data['trials_type']

        food_block = []
        water_block = [0]

        current = 'water'
        for order_i, i in enumerate(relevant_trials[1:]):
            if trials_type[i - 1] in [
                    day.ate, day.not_ate, day.omission_pellet, day.not_ate_licked, day.omission_food_taste]:
                if current == 'water':
                    food_block.append(int(iti_size * (order_i + 1)))
                    current = 'food'

            elif trials_type[i - 1] in [day.drank, day.not_drank, day.omission_water]:
                if current != 'water':
                    water_block.append(int(iti_size * (order_i + 1)))
                    current = 'water'

        return water_block, food_block

    def find_satiation_period(self, day_data, last_trial):
        relevant_trials, trials_type = day_data['relevant_trials'], day_data['trials_type']
        n_satiation_trials = len([i for i in relevant_trials if i > last_trial])
        if n_satiation_trials >= MIN_SATIATION_TRIALS:
            len_satiation = min(n_satiation_trials, len(relevant_trials) // 5)
        else:
            len_satiation = max(len(relevant_trials) // 5, MIN_SATIATION_TRIALS)

        return len_satiation

    def bin_data(self, cells, bin_size):
        cells_df = pd.DataFrame(cells)
        if bin_size == 1:
            smoothed = uniform_filter1d(cells, size=self.sec_hz, axis=1)
            return pd.DataFrame(smoothed)
        binned_data = cells_df.groupby(np.arange(cells_df.shape[1]) // bin_size, axis=1).mean()
        return binned_data
