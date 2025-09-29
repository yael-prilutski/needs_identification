import xmltodict
import numpy as np
from os import listdir
from os.path import basename, join, getsize, isfile, isdir, dirname

from configurations import *
from run import Run
from metadata import Metadata
from Processor import Processor
from main_analyzer import MainAnalyzer

DATA_DICT_NAME = 'combined_data.npy'
OPTO_DATA_DICT_NAME = 'combined_data_opto.npy'


class Day(Processor):
    def __init__(self, mouse_dict, path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = path
        self.data_dir_path = [join(self.path, file) for file in listdir(path) if 'data' in file][0]
        self.metadata = Metadata(self.data_dir_path)
        self.declare_cues()

        self.mouse_dict = mouse_dict
        self.mouse_name = basename(dirname(path))
        self.name = basename(path)

        # if self.is_light:
        #     self.data_dict_name = OPTO_DATA_DICT_NAME
        # else:
        self.data_dict_name = DATA_DICT_NAME

        self.runs_paths = [join(self.data_dir_path, file) for file in listdir(self.data_dir_path) if 'run' in file]
        self.raw_run_paths = [join(self.path, file) for file in listdir(self.path) if 'run' in file]
        self.sec_hz = self.find_sec_hz()

        self.main_analyzer = MainAnalyzer(self.sec_hz)

    @property
    def food_day(self):
        cues = self.load_data_dict()['cues']
        return self.pellet_signal in cues and self.fluid_signal not in cues

    @property
    def water_day(self):
        cues = self.load_data_dict()['cues']
        return self.fluid_signal in cues and self.pellet_signal not in cues

    @property
    def is_double(self):
        return self.metadata.is_double_frames()

    @property
    def is_light(self):
        return self.metadata.is_light()

    @property
    def is_water_ai14(self):
        return self.metadata.is_water_ai14()

    @property
    def is_omission(self):
        return self.metadata.is_omission()

    @property
    def is_water_axis(self):
        return self.metadata.is_water_axis()

    @property
    def is_food_axis(self):
        return self.metadata.is_food_axis()

    @property
    def ignore_last_day(self):
        return self.metadata.ignore_last_day()

    @property
    def negative_water_voltage(self):
        return self.metadata.negative_water_voltage()

    def suite2p_path(self):
        return [join(self.data_dir_path, d, 'plane0') for d in listdir(self.data_dir_path) if 'suite2p' in d][0]

    def combined_data_dict_path(self):
        return [join(self.data_dir_path, file)
                for file in listdir(self.data_dir_path) if self.data_dict_name in file][0]

    def declare_cues(self):
        self.pellet_signal = CUES_REGULAR['PELLET_SIGNAL']
        self.fluid_signal = CUES_REGULAR['FLUID_SIGNAL']
        self.omission_fluid_signal = CUES_REGULAR['OMISSION_FLUID_SIGNAL']
        self.omission_pellet_signal = CUES_REGULAR['OMISSION_PELLET_SIGNAL']
        self.neutral_signal = CUES_REGULAR['NEUTRAL_SIGNAL']
        self.surprise_fluid_signal = CUES_REGULAR['SURPRISE_FLUID_SIGNAL']
        self.surprise_pellet_signal = CUES_REGULAR['SURPRISE_PELLET_SIGNAL']
        self.pellet_instead_of_fluid_signal = CUES_REGULAR['PELLET_INSTEAD_OF_FLUID_SIGNAL']
        self.fluid_instead_of_pellet_signal = CUES_REGULAR['FLUID_INSTEAD_OF_PELLET_SIGNAL']

    def find_sec_hz(self):
        try:
            xml_path = [join(self.raw_run_paths[0], file)
                        for file in listdir(self.raw_run_paths[0]) if 'Experiment' in file][0]
            with open(xml_path, 'r', encoding='utf-8') as file:
                xml_output = file.read()
            xml_data = xmltodict.parse(xml_output)['ThorImageExperiment']
            frame_rate = round(float(xml_data['LSM']['@frameRate']))
            if self.is_double:
                frame_rate = round(frame_rate / 2)
            return frame_rate
        except IndexError:
            ops_path = join(self.suite2p_path(), 'ops.npy')
            return np.load(ops_path, allow_pickle=True)[()]['fs']

    def load_inputs_dict_per_run(self):
        run_dict_paths = [
            [join(run_path, file) for file in listdir(run_path)
             if 'inputs_dict' in file][0] for run_path in self.runs_paths]
        loaded_dicts = [np.load(run_dict, allow_pickle=True)[()] for run_dict in run_dict_paths]

        return loaded_dicts, run_dict_paths

    def load_data_dict(self):
        data_path = self.combined_data_dict_path()
        return np.load(data_path, allow_pickle=True).tolist()

    def find_frames_per_run(self):
        manual_count = [Run(path, self.sec_hz).manual_frames_count for path in self.runs_paths]
        frames_per_run = [0]
        for i_path, path in enumerate(self.raw_run_paths):
            xml_path = join(path, 'Experiment.xml')
            with open(xml_path, 'r', encoding='utf-8') as file:
                xml_output = file.read()
            xml_data = xmltodict.parse(xml_output)['ThorImageExperiment']

            if xml_data['ExperimentStatus']['@value'] == 'Stopped' or manual_count[i_path]:
                raw_file = join(path, 'Image_001_001.raw')
                xpx = int(xml_data['LSM']['@pixelX'])
                ypx = int(xml_data['LSM']['@pixelY'])
                frames = int(getsize(raw_file) / xpx / ypx / 2)
                frames_per_run.append(frames)
            else:
                frames_per_run.append(int(xml_data['Timelapse']['@timepoints']))

        if self.is_double:
            frames_per_run = [round(run / 2) for run in frames_per_run]

        runs_indexes = [sum(frames_per_run[:i + 1]) for i in range(len(frames_per_run))]
        return runs_indexes

    def load_suite2p_output(self, file_types):
        suite2p_dir_path = self.suite2p_path()
        files_to_load = [join(suite2p_dir_path, file) for file in file_types]
        return [np.load(path, allow_pickle=True) for path in files_to_load]
