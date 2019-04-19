# encoding:utf-8

import inspect


def get_runtime_function_name():
    return inspect.stack()[1][3]
