# encoding:utf-8

import sys
import inspect


def get_runtime_function_name():
    return inspect.stack()[1][3]


def to_str(string, encoding="utf-8"):
    """convert to str for print"""
    if sys.version_info.major == 3:
        if isinstance(string, bytes):
            return string.decode(encoding)
    elif sys.version_info.major == 2:
        if isinstance(string, unicode):
            return string.encode(encoding)
    return string
