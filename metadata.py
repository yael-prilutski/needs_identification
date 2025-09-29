from os import listdir


class Metadata:

    class Flags:

        REGULAR_DAY = 'regular'
        LIGHT_DAY = 'light'
        DOUBLE_FRAMES = 'double'
        NO_CAMERA = 'nocamera'
        NO_CUES = 'nocues'
        SHORT_SYNC = 'shortsync'
        WATER_AI14 = 'waterai14'
        OMISSION = 'omission'
        WATER_AXIS = 'wateraxis'
        FOOD_AXIS = 'foodaxis'
        IGNORE_LAST_DAY = 'ignorelastday'
        NEGATIVE_WATER_VOLTAGE = 'vwnegative'
        NO_DEEPLABCUT = 'nodeeplabcut'
        NO_PELLET_LICKS = 'nopelletlicks'
        CONSTANT_PELLET_LICKS = 'constantpelletlicks'
        NO_BB_DOWN = 'nobbdown'
        ROI_STUCK_CAP = 'roistuckcap'
        NO_INCREASES = 'noincreases'
        USE_PELLET_X = 'usepelletx'
        IGNORE_ROI = 'ignoreroi'
        MANUAL_FRAMES = 'manualframescount'

    def __init__(self, folder_path, default=Flags.REGULAR_DAY):
        self._meta_props = (_s := set(f.lower().strip('_') for f in listdir(folder_path) if f.startswith('_')))
        _s or _s.add(default)

    @property
    def tags(self): return sorted(list(self._meta_props))

    def has_flag(self, flag): return flag in self._meta_props
    def is_regular(self): return self.has_flag(self.Flags.REGULAR_DAY)
    def is_light(self): return self.has_flag(self.Flags.LIGHT_DAY)

    def is_double_frames(self): return self.has_flag(self.Flags.DOUBLE_FRAMES)

    def no_camera(self): return self.has_flag(self.Flags.NO_CAMERA)

    def no_deeplabcut(self): return self.has_flag(self.Flags.NO_DEEPLABCUT)

    def no_cues(self): return self.has_flag(self.Flags.NO_CUES)

    def short_sync(self): return self.has_flag(self.Flags.SHORT_SYNC)

    def is_water_ai14(self): return self.has_flag(self.Flags.WATER_AI14)

    def is_omission(self): return self.has_flag(self.Flags.OMISSION)

    def is_water_axis(self): return self.has_flag(self.Flags.WATER_AXIS)

    def is_food_axis(self): return self.has_flag(self.Flags.FOOD_AXIS)

    def ignore_last_day(self): return self.has_flag(self.Flags.IGNORE_LAST_DAY)

    def negative_water_voltage(self): return self.has_flag(self.Flags.NEGATIVE_WATER_VOLTAGE)

    def no_pellet_licks(self): return self.has_flag(self.Flags.NO_PELLET_LICKS)

    def constant_pellet_licks(self): return self.has_flag(self.Flags.CONSTANT_PELLET_LICKS)

    def no_bb_down(self): return self.has_flag(self.Flags.NO_BB_DOWN)

    def roi_stuck_cap(self): return self.has_flag(self.Flags.ROI_STUCK_CAP)

    def no_increases(self): return self.has_flag(self.Flags.NO_INCREASES)

    def use_pellet_x(self): return self.has_flag(self.Flags.USE_PELLET_X)

    def ignore_roi(self): return self.has_flag(self.Flags.IGNORE_ROI)

    def manual_frames_count(self): return self.has_flag(self.Flags.MANUAL_FRAMES)
