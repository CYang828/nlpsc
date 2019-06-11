# encoding:utf-8

import random

colors = ['#FF0F00', '#FF6600', '#FF9E01', '#FCD202', '#F8FF01',
          '#04D215', '#0D8ECF', '#0D52D1', '#2A0CD0', '#8A0CCF', '#CD0D74']


def random_color():
    l = len(colors)
    random.seed(l)
    i = random.randint(0, l-1)
    return colors[i]


def yield_color(n=1):
    l = len(colors)
    idx = n % l
    return colors[idx]





