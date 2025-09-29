import glob
from os import listdir
from os.path import basename, join, dirname, isdir

from day import Day
from week import Week
from main_analyzer import MainAnalyzer


class Mouse(object):

    def __init__(self, mouse_dict, days, week=None, location=None):

        self.path = mouse_dict['path']
        self.name = basename(mouse_dict['path'])
        self.days_paths = self.find_days(days)
        self.days = [Day(mouse_dict, i) for i in self.days_paths]
        self.sec_hz = self.days[0].sec_hz
        if week:
            self.week = Week(self.path, week, self.days, self.name)
        if location:
            self.location = location
        self.main_analyzer = MainAnalyzer(self.days[0].sec_hz)

    def find_days(self, days_list):
        all_files = listdir(self.path)
        days_paths = [join(self.path, day) for day in days_list if day in all_files]
        assert len(days_list) == len(days_paths)
        return days_paths
