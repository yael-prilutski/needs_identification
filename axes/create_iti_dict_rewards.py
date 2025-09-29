import numpy as np
from os.path import join

from configurations import *
from mouse import Mouse
from Processor import Processor
from axes_analyzer import AxesAnalyzer

RELEVANT_DAYS = 'week1_days'
WEEK = 'week1'
RELEVANT_MICE = ALL_MICE


class CreateITIDictRewards(Processor):

    def __init__(self, mouse, *args, **kwargs):
        Processor.__init__(self, *args, **kwargs)
        self.mouse = mouse
        self.analyzer = AxesAnalyzer(mouse.smooth_factor)

    def run(self):
        iti_dict = {}
        for day in self.mouse.days[:4]:
            iti_df, relevant_trials, relevant_cues = self.analyzer.find_iti_df(day, reward_trials=True)
            day_dict = {
                'iti_df': iti_df,
                'relevant_trials': relevant_trials,
                'trials_classification': relevant_cues,
                'size_iti': self.analyzer.iti_slice,
            }
            iti_dict[day.name] = day_dict
            print(f'finished processing {day.name}')

        np.save(join(self.mouse.week.data_dir_path, 'axes', f'{self.mouse.name}_iti_dict_rewards.npy'), iti_dict)


def main():
    for mouse_path in RELEVANT_MICE:
        if WEEK in mouse_path.keys():
            mouse = Mouse(mouse_path, mouse_path[RELEVANT_DAYS], mouse_path[WEEK])
            process = CreateITIDictRewards(mouse)
            process.run()
            print(f'finished processing {mouse.name}')


'__main__' == __name__ and main()
