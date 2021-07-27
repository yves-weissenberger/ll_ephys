import os
import sys
import re
import datetime
from datetime import datetime as dt


def get_video_timestamp(fname):
    """ e.g. _2021-07-25-120729"""
    date_string = re.findall(r'_([0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{6})',fname)[0]
    date_object = dt.strptime('2021-07-25-120729','%Y-%M-%d-%H%m%S')
    return date_object