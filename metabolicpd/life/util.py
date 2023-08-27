# -*- coding: utf-8 -*-

import numpy as np


def hill(x, p=1, k=1.0):
    """x: positive mass of tail node, p: power (int), k some 'dissociation' constant."""
    x_p = np.power(x, p)
    return np.divide(x_p, k + x_p)


def min(x, i):
    return np.min(x[i])
