import glob
import numpy as np
from os import mkdir
from os.path import join, basename, dirname, isdir
from scipy import io
from sklearn.preprocessing import MinMaxScaler

from Processor import Processor
from metadata import Metadata


class Run(Processor):

    def __init__(self, path, sec_hz, is_light=False, diff_water=False, negative_water_voltage=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = Metadata(path)
        self.is_light = is_light
        self.diff_water = diff_water
        self.negative_water_voltage = negative_water_voltage
        self.name = basename(path)
        self.sec_hz = sec_hz
        self.path = path
        self.cues = self._find_cues()

    def _find_cues(self):
        if self.no_cues:
            return []
        else:
            mat_file = glob.glob(self.path + "/**/*.mat", recursive=True)[0]
            mat_data = io.loadmat(mat_file)
            try:
                cues = mat_data['TrialRecord'][0][0][7][0]
            except KeyError:
                cues = [mat_data[f'Trial{i}'][0][0][4][0][0] for i in range(1, len(mat_data.keys()) - 5)][:-1]
            return cues

    @property
    def space_between_cues(self):
        space = self.sec_hz * self.sec_space_between_cues
        if self.is_light:
            space = 0
        return space

    @property
    def no_cues(self):
        return self.metadata.no_cues()

    @property
    def short_sync(self):
        return self.metadata.short_sync()

    @property
    def no_camera(self):
        return self.metadata.no_camera()

    @property
    def do_not_use_deeplabcut(self):
        return self.metadata.no_deeplabcut()

    @property
    def no_pellet_licks(self):
        return self.metadata.no_pellet_licks()

    @property
    def constant_pellet_licks(self):
        return self.metadata.constant_pellet_licks()

    @property
    def no_bb_down(self):
        return self.metadata.no_bb_down()

    @property
    def roi_stuck_cap(self):
        return self.metadata.roi_stuck_cap()

    @property
    def no_increases(self):
        return self.metadata.no_increases()

    @property
    def use_pellet_x(self):
        return self.metadata.use_pellet_x()

    @property
    def ignore_roi(self):
        return self.metadata.ignore_roi()

    @property
    def manual_frames_count(self):
        return self.metadata.manual_frames_count()

    def save_behavioral_plots(self, data_dict, cues_trials, frames):
        fig_location = join(dirname(self.path), 'figures')
        trials_types_dir = join(fig_location, f'trials_classification_{self.name[-4:]}')

        isdir(fig_location) or mkdir(fig_location)
        isdir(trials_types_dir) or mkdir(trials_types_dir)

        fig_path = join(trials_types_dir, f'{self.name}_inputs.jpg')

        cues_array = np.concatenate(cues_trials)
        keys = ['licks_water', 'licks_pellet', 'pellet_y', 'bb_up', 'bb_down', 'pellet_img']

        # Initialize empty lists to store concatenated arrays
        concatenated_arrays = []

        # Iterate over the keys
        for key in keys:
            try:
                concatenated_array = np.concatenate(data_dict[key])
                concatenated_arrays.append(concatenated_array)
            except (ValueError, KeyError):
                concatenated_arrays.append([])

        water_licks, pellet_licks, pellet_y, bb_up, bb_down, pellet_img = concatenated_arrays
        scaler = MinMaxScaler()
        try:
            pellet_y = scaler.fit_transform(np.array(pellet_y).reshape(-1, 1))
        except ValueError:
            pellet_y = [[0] for _ in range(len(water_licks))]

        self.visualizer.inputs_representation_plots(
            water_licks, pellet_licks, bb_up, bb_down, pellet_img, pellet_y, self.name, fig_path, frames, cues_array)
        self.visualizer.show_inputs_by_trials(data_dict, trials_types_dir, cues_trials)

    def save_behavioral_dict(self, data_dict):
        data_dict.pop('frames', 'not found')
        data_dict.pop('cues_trials', 'not found')
        np.save(join(self.path, f'{self.name}_inputs_dict.npy'), data_dict)
        print(f'saved behavioral dict of {self.path}')
