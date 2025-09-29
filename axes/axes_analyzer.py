import numpy as np
import pandas as pd
from os.path import isfile, join

from Processor import Processor
from main_analyzer import MainAnalyzer
from run import Run


class AxesAnalyzer(Processor):

    def __init__(self, sec_hz, *args, **kwargs):
        Processor.__init__(self, *args, **kwargs)
        self.sec_hz = sec_hz
        self.trials_limit_for_axes = 30
        self.main_analyzer = MainAnalyzer(sec_hz)

    @property
    def iti_slice(self):
        return 3 * self.sec_hz

    def find_last_reward(self, classification, cues, day):
        classification = np.array(classification)
        max_no_consumption = 5
        if day.pellet_signal in cues and day.fluid_signal in cues:
            all_rewards_trials = np.where(
                (classification == day.ate) | (classification == day.omission_food_taste) | (classification == day.drank))[0]
            return all_rewards_trials[-1], all_rewards_trials
        else:
            if day.pellet_signal in cues:
                trials_for_consumption_index = np.where(
                    (classification == day.ate) | (classification == day.omission_food_taste) | (
                            classification == day.problem))[0]
                all_rewards_trials = np.where(
                    (classification == day.ate) | (classification == day.omission_food_taste))[0]
                all_cues = np.where(np.array(cues) == day.pellet_signal)[0]
            else:
                trials_for_consumption_index = np.where((classification == day.drank) | (classification == day.problem))[0]
                all_rewards_trials = np.where(classification == day.drank)[0]
                all_cues = np.where(np.array(cues) == day.fluid_signal)[0]
                if day.mouse_name == 'YP79' and day.drank in classification:
                    trials_for_consumption_index = trials_for_consumption_index[:150]
                    all_rewards_trials = all_rewards_trials[:150]
            consumption_index = [True if i in trials_for_consumption_index else False for i in all_cues]
            last_reward = all_rewards_trials[-1]
            for i in range(len(consumption_index) - max_no_consumption):
                if consumption_index[i:i + max_no_consumption] == [False] * max_no_consumption:
                    last_reward = all_cues[i]
                    if all_cues[i] not in all_rewards_trials:
                        last_reward = [t for t in all_rewards_trials if t < all_cues[i]][-1]
                    break
            return last_reward, all_rewards_trials

    def find_relevant_neutral_index(self, day, classification, licks_water, licks_food):
        ignore_pellet_licks = sum([Run(path, self.sec_hz).constant_pellet_licks for path in day.runs_paths])
        neutral_index = np.where(classification == day.neutral)[0]

        good_neutral_trials_index = []
        for i in neutral_index:
            try:
                water_licks = licks_water[i][-self.iti_slice * 2:]
            except IndexError:
                water_licks = [0]
            food_licks = [0]
            if not ignore_pellet_licks:
                try:
                    food_licks = licks_food[i][-self.iti_slice * 2:]
                except IndexError:
                    pass
            if sum(water_licks) == 0 and sum(food_licks) == 0:
                good_neutral_trials_index.append(i)

        return good_neutral_trials_index

    def find_trials_for_axes(self, day, cues, classification, good_neutral_trials_index):
        last_reward, all_rewards_trials = self.find_last_reward(classification, cues, day)
        try:
            limit_reward_start_index = min(all_rewards_trials[int(len(all_rewards_trials) * 0.2)],
                                           all_rewards_trials[self.trials_limit_for_axes])
        except Exception:
            limit_reward_start_index = all_rewards_trials[int(len(all_rewards_trials) * 0.2)]
        # if day.is_light:
        #     light_start = np.where(np.array(cues) == day.opto_trigger_signal)[0][0]
        #     if light_start < limit_reward_start_index:
        #         limit_reward_start_index = light_start
        limit_reward_end_index = last_reward
        start_index = [i for i in good_neutral_trials_index if i < limit_reward_start_index]
        end_index = [i for i in good_neutral_trials_index if i < limit_reward_end_index][-len(start_index):]

        return start_index, end_index

    def select_iti(self, cell_data, relevant_trials_index):
        iti_ends = []
        for trial in relevant_trials_index:
            iti_ends.extend(cell_data[trial][-self.iti_slice:])
        return np.array(iti_ends)

    def cut_iti(self, cells_data, index, do_mean=False):
        cells_iti = [self.select_iti(cells_data[cell], index) for cell in range(len(cells_data))]
        if do_mean:
            return np.array([cell.mean() for cell in cells_iti])
        else:
            return cells_iti

    def create_dot_product_iti(self, cells_iti_df, vector):
        start_vector = vector['start']
        end_vector = vector['end']
        sub_start_finish = vector['sub']
        # dot_products = []
        # for frame in range(len(cells_iti_df.iloc[0])):
        #     dot_product = np.dot(cells_iti_df.iloc[:, frame], sub_start_finish)
        #     product = (dot_product - end_vector) / (start_vector - end_vector)
        #     dot_products.append(product)

        dot_products2 = cells_iti_df.T @ sub_start_finish
        normalized_dot_products = (dot_products2 - end_vector) / (start_vector - end_vector)

        return normalized_dot_products

    def find_axes(self, day):
        day_data = day.load_data_dict()
        day_cues = day_data['cues']
        day_trials = day_data['trials_classification'][:-1]
        if day.is_light:
            day_cues = self.main_analyzer.change_to_regular_opto_cues(day, day_cues)
            day_trials = self.main_analyzer.change_to_regular_opto_trials(day, day_trials)

        relevant_trials_index = self.find_relevant_neutral_index(
            day, day_trials, day_data['water_licks'], day_data['pellet_licks'])
        start_index, end_index = self.find_trials_for_axes(day, day_cues, day_trials, relevant_trials_index)

        fix_cells_baseline = [[np.hstack([cell[trial][self.sec_hz * 2:], cell[trial + 1][:self.sec_hz * 2]])
                               for trial in range(len(cell) - 1)] for cell in day_data['cells']]
        day_cells_data = self.main_analyzer.min_max_normalization(fix_cells_baseline)
        start_iti = self.cut_iti(day_cells_data, start_index, do_mean=True)
        end_iti = self.cut_iti(day_cells_data, end_index, do_mean=True)

        return start_iti, end_iti, end_index[-1]

    def find_relevant_reward_index(self, day, cues, classification, licks_water, licks_food, increases):
        ignore_pellet_licks = sum([Run(path, self.sec_hz).constant_pellet_licks for path in day.runs_paths])
        reward_index = np.where((cues == day.fluid_signal) | (cues == day.pellet_signal))[0]

        check_increases = len(increases) > 0

        good_reward_trials_index = []
        for i in reward_index:
            trial_food_licks = [0]
            trial_increase = None
            if classification[i] == day.problem:
                continue
            trial_water_licks = licks_water[i][-self.iti_slice * 2:]
            if not ignore_pellet_licks:
                try:
                    trial_food_licks = licks_food[i][-self.iti_slice * 2:]
                except IndexError:
                    pass
            if check_increases:
                if increases[i] and increases[i] > len(licks_food[i]) - self.iti_slice * 2:
                    trial_increase = increases[i]
            if sum(trial_water_licks) == 0 and sum(trial_food_licks) == 0 and not trial_increase:
                good_reward_trials_index.append(i)

        return good_reward_trials_index

    def find_iti_df(self, day, reward_trials=False):
        day_data = day.load_data_dict()
        day_trials = day_data['trials_classification'][:-1]
        if day.is_light:
            day_trials = self.main_analyzer.change_to_regular_opto_trials(day, day_trials)

        if reward_trials:
            relevant_trials_index = self.find_relevant_reward_index(
                day, day_data['cues'][:-1], day_trials, day_data['water_licks'], day_data['pellet_licks'],
                day_data['increases'])
        else:
            relevant_trials_index = self.find_relevant_neutral_index(
                day, day_trials, day_data['water_licks'], day_data['pellet_licks'])

        fix_cells_baseline = self.main_analyzer.rearrange_cells_without_baseline(day_data['cells'])
        day_cells_data = self.main_analyzer.min_max_normalization(fix_cells_baseline)
        cells_iti = self.cut_iti(day_cells_data, relevant_trials_index)

        return pd.DataFrame(cells_iti), relevant_trials_index, day_trials

    def find_vector(self, day):
        start_of_session, end_of_session, last_index = self.find_axes(day)

        sub_start_finish = end_of_session - start_of_session
        start_vector = np.dot(start_of_session, sub_start_finish)
        end_vector = np.dot(end_of_session, sub_start_finish)

        return sub_start_finish, start_vector, end_vector, start_of_session, end_of_session, last_index

    def classify_by_behavior_colors(self, day, day_data, length_day, iti_size, before_behavior=True, reward_trials=False):
        last_trial = day_data['last_vector_trial']
        relevant_trials = [i for i in day_data['relevant_trials'] if i < last_trial]
        trials_type = day_data['trials_type']
        colors = []
        for i in relevant_trials:
            if before_behavior:
                if reward_trials:
                    trial = trials_type[i]
                else:
                    trial = trials_type[i - 1]
            else:
                trial = trials_type[i + 1]

            if before_behavior and i == 0 and not reward_trials:
                colors.extend(['lightgray'] * iti_size)
            elif not before_behavior and i == len(relevant_trials) - 1:
                colors.extend(['lightgray'] * iti_size)
            elif trial == day.neutral:
                colors.extend(['lightgray'] * iti_size)
            elif trial == day.drank:
                colors.extend(['#A9BDD8'] * iti_size)
            elif trial == day.not_drank:
                colors.extend(['#396386'] * iti_size)
            elif trial == day.omission_water:
                colors.extend(['green'] * iti_size)
            elif trial in [day.ate, day.omission_food_taste]:
                colors.extend(['#A52B17'] * iti_size)
            elif trial in [day.not_ate, day.not_ate_licked]:
                colors.extend(['#866446'] * iti_size)
            elif trial == day.omission_pellet:
                colors.extend(['purple'] * iti_size)
            else:
                colors.extend(['black'] * iti_size)
        assert len(colors) == length_day
        return colors
