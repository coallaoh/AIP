#!/usr/bin/env python

__author__ = 'joon'

import nash
import numpy as np

p = np.array([
    [4.0, 6.6, 15.0, 22.2, 16.7, 9.9],
    [2.5, 2.3, 11.6, 18.5, 7.2, 4.9],
    [5.8, 7.6, 4.6, 23.6, 16.6, 9.1],
    [0.4, 0.8, 8.6, 5.8, 3.1, 1.4],
    [2.6, 2.2, 11.8, 18.1, 3.4, 4.3],
    [0.7, 0.9, 5.2, 9.5, 3.2, 2.0]
])

minimaxgame = nash.Game(-p)
eq = [e for e in minimaxgame.vertex_enumeration()]
roweq = eq[0][0]
coleq = eq[0][1]
value = np.dot(roweq, np.dot(p, coleq))

print("Value of the game (privacy guarantee for U): %2.1f" % value)
print("U's optimal strategy: {}".format(roweq))
print("R's optimal strategy: {}".format(coleq))
