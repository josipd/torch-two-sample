from .statistics_diff import (
    SmoothFRStatistic, SmoothKNNStatistic, MMDStatistic, EnergyStatistic)
from .statistics_nondiff import FRStatistic, KNNStatistic

__all__ = ['SmoothFRStatistic', 'SmoothKNNStatistic', 'MMDStatistic',
           'EnergyStatistic', 'FRStatistic', 'KNNStatistic']
