import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ========
# Appendix
# ========

# fixed various very small issues in file_2.py

# get index and values from a collection
my_collection = ['a', 'b', 'c']
for i, value in enumerate(my_collection):
    print(i, my_collection[i])

# convert an iterable object into a list
x = None
if not isinstance(x, list) and np.isiterable(x):
    x = list(x)

# create a dict from a sequence
some_list = ['foo', 'bar', 'baz']
mapping = dict((v, i) for i, v in enumerate(some_list))

z = mapping

# get a sorted list of unique elements in a sequence
sorted(set('this is just some string'))

# create list of pairs from two lists
seq1 = ['foo', 'bar', 'baz']
seq2 = ['one', 'two', 'three']
zip(seq1, seq2)

# TODO this feature should be implemented
