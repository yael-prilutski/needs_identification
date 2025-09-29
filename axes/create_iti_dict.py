import numpy as np
from os.path import join
from os import makedirs

from configurations import *
from mouse import Mouse
from Processor import Processor
from axes_analyzer import AxesAnalyzer

WEEK = 'week2'
RELEVANT_DAYS = f'{WEEK}_days'
# REWARD = 'food'
# RELEVANT_DAYS = f'opto_{REWARD}_days'
# WEEK = f'opto_{REWARD}_week'
RELEVANT_MICE = ALL_MICE


class CreateITIDict(Processor):

    def __init__(self, mouse, *args, **kwargs):
        Processor.__init__(self, *args, **kwargs)
        self.mouse = mouse
        self.analyzer = AxesAnalyzer(mouse.smooth_factor)

    def run(self):
        iti_dict = {}
        for day in self.mouse.days:
            iti_df, relevant_trials, relevant_cues = self.analyzer.find_iti_df(day)
            day_dict = {
                'iti_df': iti_df,
                'relevant_trials': relevant_trials,
                'trials_classification': relevant_cues,
                'size_iti': self.analyzer.iti_slice,
            }
            iti_dict[day.name] = day_dict

        makedirs(join(self.mouse.week.data_dir_path, 'axes'), exist_ok=True)
        np.save(join(self.mouse.week.data_dir_path, 'axes', f'{self.mouse.name}_iti_dict.npy'), iti_dict)


def main():
    for mouse_path in RELEVANT_MICE:
        if WEEK in mouse_path.keys():
            mouse = Mouse(mouse_path, mouse_path[RELEVANT_DAYS], mouse_path[WEEK])
            process = CreateITIDict(mouse)
            process.run()
            print(f'finished processing {mouse.name}')


'__main__' == __name__ and main()
