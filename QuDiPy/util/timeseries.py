""" Custom Iterable Container to Hold Time Series Data
Written by: Amro Dodin (Willard Group - MIT)

"""

import numpy as np


class TimeSeries:
    def append(self, data, time):
        self.data.append(data)
        self.time.append(time)
        self.length += 1

    def __iter__(self):
        return zip(self.data, self.time)

    def __eq__(self, other):
        if len(self.data) > 0 and hasattr(self.data[0], 'round'):
            self_data_rounded = np.round(self.data, 7)
            other_data_rounded = np.round(other.data, 7)
            data_equality = np.array_equal(self_data_rounded, other_data_rounded)
        else:
            data_equality = np.array_equal(self.data, other.data)
        self_time_rounded = np.round(self.time, 7)
        other_time_rounded = np.round(other.time, 7)
        time_equality = np.array_equal(self_time_rounded, other_time_rounded)
        return data_equality and time_equality

    def __init__(self):
        self.data = []
        self.time = []
        self.length = 0

        self.__index = 0
