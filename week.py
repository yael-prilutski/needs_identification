from os import listdir
from os.path import join

from day import Day


class Week(Day):
    def __init__(self, mouse_path, week_name, days, *args, **kwargs):
        self.path = join(mouse_path, week_name)
        super().__init__(mouse_path, self.path, *args, **kwargs)
        self.days = days
        self.declare_cues()
