from random import choice

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ========
# Appendix
# ========

# a few more comments will make it more readable
# comment 1
# comment 2
# comment 3


# the line below has been updated to fix the bug
person_123 = choice(['a', 'v', 'c'])

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

# get a sorted list of unique elements in a sequence
sorted(set('this is just some string'))

# create list of pairs from two lists
seq1 = ['foo', 'bar', 'baz']
seq2 = ['one', 'two', 'three']
zip(seq1, seq2)

# simultaneously iterating through two lists
for i, (a, b) in enumerate(zip(seq1, seq2)):
    print('%d: %s, %s' % (i, a, b))

# getting keys ans values from a dict
d1 = {'a': 'some value', 'b': [1, 2, 3, 4]}
d1.keys()
d1.values()

# create a dict from a list of couples
list1 = range(5)
list2 = reversed(range(5))
list_couples = zip(list1, list2)
mapping = dict(list_couples)

# list comprehension : creating a list by transforming a subset of a list
strings = ['a', 'as', 'bat', 'car', 'dove', 'python']
[x.upper() for x in strings if len(x) > 2]


# defining a function
def my_function(x, y, z=1.5):
    if z > 1:
        return z * (x + y)
    else:
        return z / (x + y)


# use global variable in a function
a = None


def bind_a_variable():
    global a
    a = 3


bind_a_variable()
print(a)


# a function that returns several variables
def f():
    a = 5
    b = 6
    c = 7
    return a, b, c


a, b, c = f()


# getting args and kwargs from the arguments list in a function
def say_hello_then_call_f(f, *args, **kwargs):
    print('args is', args)
    print('kwargs is', kwargs)
    print("Hello! Now I'm going to call %s" % f)
    return f(*args, **kwargs)


def g(x, y, z=1):
    return (x + y) / z


say_hello_then_call_f(g, 1, 2, z=5.)

# basic file reading without pandas
path = '/Users/pirminlemberger/PycharmProjects/PythonBook/text files/segismundo.txt'
f = open(path)
lines = [x.rstrip() for x in open(path)]

# ================
# chapitre 4 NumPy
# ================

# shape and dtype of a ndarray
data = np.array([[0.9526, -0.246, -0.8856], [0.5639, 0.2379, 0.9104]])
data.dtype
data.shape

# creating an array with zeros or ones
np.zeros((2, 3))
np.ones((3, 4))

# creating an array with a sequence of integers
arr = np.arange(10)

# indexing 2 dimensional arrays uses matrix conventions.
# axis 0 numbers lines, axis 1 numbers columns
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[0][2]
arr2d[0, 2]

# generating a array of random numbers
data = np.random.rand(7, 4)

# selecting data using boolean indexing
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data[names == 'Bob']

mask = (names == 'Bob') | (names == 'Will')
data[mask]

# universal functions for numpy arrays
np.sqrt(arr)
np.exp(arr)

# statistical functions for numpy arrays
arr = np.random.randn(5, 4)
arr.mean()
np.mean(arr)
arr.mean(axis=1)
arr.sum(0)

# sorting numpy arrays
arr = np.random.randn(8)
arr.sort()

# getting unique values from an array
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names)
pd.Series(names).unique()

# matrix multiplication
x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
p1 = x.dot(y)
p2 = np.dot(x, y)

# generating random normal sample
samples = np.random.normal(size=(4, 4))

# caster, conversion d'une liste
list = [1, 2, 3]
np.array(list, dtype=np.str)

# =================
# chapitre 5 Pandas
# =================

# definition of a Series and getting values and index
obj = pd.Series([4, 7, -5, 3])
obj.values
obj.index

# creating a Series with an index
obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])

# creating a Series from a dict
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = pd.Series(sdata)

# testing for nullity in a Series
states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = pd.Series(sdata, index=states)
pd.isnull(obj4)

# example DataFrame, see Table 5-1 for an exhaustive list of DataFrame constructors
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = pd.DataFrame(data)

# getting the data in a DataFrame as a NumPy array
data.values()

# ordering columns of a DataFrame
pd.DataFrame(data, columns=['year', 'state', 'pop'])

# retrieving a column from a DataFrame as a Series
frame2 = pd.DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
                      index=['one', 'two', 'three', 'four', 'five'])
frame2['state']
frame2.year

# retrieving a row from a DataFrame by its index
frame2.ix['three']

# adding a new column to a DataFrame
frame2['eastern'] = frame2.state == 'Ohio'

# transposing columns and rows
frame2.T

# giving names to the lists of rows and columns
pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3 = pd.DataFrame(pop)
frame3.index.name = 'year';
frame3.columns.name = 'state'

# reordering with a new index
obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])

# dropping, deleting entries from a Series
obj = pd.Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
new_obj = obj.drop('c')

# dropping lines from a DataFrame
data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=['Ohio', 'Colorado', 'Utah', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
data.drop(['Colorado', 'Ohio'], axis=0)

# dropping columns from a DataFrame
data.drop('two', axis=1)

# (1) selecting rows in a DataFrame using a boolean array
data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=['Ohio', 'Colorado', 'Utah', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
data[data['three'] > 5]

# (2) selecting elements in a DataFrame using a boolean array
data[data < 5]

# apply an array function to each column of a DataFrame
data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=['Ohio', 'Colorado', 'Utah', 'New York'],
                    columns=['one', 'two', 'three', 'four'])


def f(x):
    return x.max()


data.apply(f, axis=0)

# apply an array function to each row of a DataFrame
data.apply(f, axis=1)

# apply a function to each element of a Series
series = pd.Series([4, 2, 3])


def g(x):
    return x ** 2


series.map(g)

# apply a function to each element of a DataFrame
data.applymap(g)

# sorting a Series by its index
obj = pd.Series(range(4), index=['d', 'a', 'b', 'c'])
obj.sort_index()

# sorting a Series by its values
obj = pd.Series([4, 7, -3, 2])
obj.order()

# summing the values in a Series
obj.sum()

# summing the values in a DataFrame by rows
data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=['Ohio', 'Colorado', 'Utah', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
data.sum(axis=0)

# summing the values in a DataFrame by columns
data.sum(axis=1)

# summary of various statistic in a DataFrame
data.describe()

# computing variance and correlations matrices
data = pd.DataFrame(np.random.rand(4, 4))
data.corr()
data.cov()

# retrieving unique values from a Series or a List
obj = pd.Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
uniques = obj.unique()

# retrieving value frequencies from a Series
obj.value_counts()
pd.value_counts(obj.values, sort=False)

# using columns of a DataFrame as an index
frame = pd.DataFrame({'a': range(7), 'b': range(7, 0, -1),
                      'c': ['one', 'one', 'one', 'two', 'two', 'two', 'two'],
                      'd': [0, 1, 2, 0, 1, 2, 3]})
frame2 = frame.set_index(['c', 'd'])
frame3 = frame.set_index(['c', 'd'], drop=False)

# removing an index from a DataFrame to put it in as new columns
frame4 = frame2.reset_index()

# ======================
# chapter 6 Data Loading
# ======================

# reading data from a CSV file
df1 = pd.read_csv('./text files/ch06/ex1.csv')
df2 = pd.read_table('./text files/ch06/ex1.csv', sep=',')

# reading data from a CSV file and defining column headers
df3 = pd.read_csv('./text files/ch06/ex2.csv', names=['a', 'b', 'c', 'd', 'message'])

# reading data from a CSV file and defining an index
df4 = pd.read_csv('./text files/ch06/ex2.csv', names=['a', 'b', 'c', 'd', 'message'], index_col='message')

# print out the first n row of a DataFrame
df1.head(3)

# print out the last n rows of a DataFrame
df1.tail(3)

# read a large text file in chunks
chunker = pd.read_csv('ch06/ex6.csv', chunksize=1000)

# aggregating the pieces from a chunker
tot = pd.Series([])
for piece in chunker:
    tot = tot.add(piece['key'].value_counts(), fill_value=0)
tot = tot.order(ascending=False)

# writing data
data.to_csv('ch06/out.csv')

# reading data from an Excel file
xls_file = pd.ExcelFile('data.xls')

# ========================
# chapter 7 Data Wrangling
# ========================

# merging two DataFrames on a key with a different name in each frame
df1 = pd.DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})
df2 = pd.DataFrame({'rkey': ['a', 'b', 'd'], 'data2': range(3)})
df3 = pd.merge(df1, df2, left_on='lkey', right_on='rkey')

# merging using an outer join and an index
left1 = pd.DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'], 'value': range(6)})
right1 = pd.DataFrame({'group_val': [3.5, 7]}, index=['a', 'b'])
df = pd.merge(left1, right1, left_on='key', right_index=True, how='outer')

# concatenate two NumPy arrays horizontally
arr1 = np.arange(12).reshape((3, 4))
arr2 = arr1 + 100
arr3 = np.concatenate([arr1, arr2], axis=1)

# concatenate two NumPy arrays vertically
arr4 = np.concatenate([arr1, arr2], axis=0)

# concatenate several pandas Series vertically to form a new Series
s1 = pd.Series([0, 1], index=['a', 'b'])
s2 = pd.Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = pd.Series([5, 6], index=['f', 'g'])
result = pd.concat([s1, s2, s3], axis=0)

# concatenate several pandas Series horizonatlly to form a DataFrame
result = pd.concat([s1, s2, s3], axis=1)

# testing for duplicated lines in a DataFrame
data = pd.DataFrame({'k1': ['one'] * 3 + ['two'] * 4, 'k2': [1, 1, 2, 3, 3, 4, 4]})
data.duplicated()

# droping duplicated lines from a DataFrame
cleaned = data.drop_duplicates()

# droping duplicated lines from a DataFrame based on specific columns
data = pd.DataFrame({'k1': ['one'] * 3 + ['two'] * 4, 'k2': [1, 1, 2, 3, 3, 4, 4]})
data['v1'] = range(7)
cleaned = data.drop_duplicates(['k1', 'k2'])

# replacing values in a Series
data = pd.Series([1., -999., 2., -999., -1000., 3.])
cleaned = data.replace([-999, -1000], [np.nan, 0])

# discretization and binning
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]
cats = pd.cut(ages, bins)
labels = cats.codes
nb_elements_in_bins = pd.value_counts(cats)  # le résultat qu'on cherche

# discritization using quantiles
data = np.random.randn(1000)  # Normally distributed
cats = pd.qcut(data, 4)  # Cut into quartiles
nb_elements_in_bins = pd.value_counts(cats)

# filtering outliers in one column of a DataFrame
np.random.seed(12345)
data = pd.DataFrame(np.random.randn(1000, 4))
col = data[3]
data[np.abs(col) > 3]

# filtering outliers in any column of a DataFrame
data[(np.abs(data) > 3).any(1)]

# shuffle the rows of a DataFrame
df = pd.DataFrame(np.arange(10 * 4).reshape(10, 4))
sampler = np.random.permutation(10)
df_shuffled = df.ix[sampler]

# select a fixed size sample from a DataFrame without replacement
df.take(np.random.permutation(len(df))[:3])

# select a fixed size sample from a DataFrame with replacement
bag = np.array([5, 7, -1, 6, 4])
sampler = np.random.randint(0, len(bag), size=10)
draws = bag.take(sampler)

# dummyfication of categorical columns
df = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'], 'data1': range(6)})
dummies = pd.get_dummies(df['key'], prefix='key')
df_with_dummy = df[['data1']].join(dummies)

# finding a substring in a string
val = 'a,b, guido'
answer = 'guido' in val
location = val.index('guido')
nb = val.count(',')

# pivoting, replacing columns with lines
data = pd.DataFrame(np.arange(6).reshape((2, 3)),
                    index=pd.Index(['Ohio', 'Colorado'], name='state'),
                    columns=pd.Index(['one', 'two', 'three'], name='number'))

result = data.stack()
result.unstack()

# ====================================
# chapter 8 Plotting and Visualization
# ====================================

# testing wether pyplot works properly (look for hidden windows!)
plt.plot(np.arange(10))

# plotting with low level matplotlib API
# --------------------------------------
fig = plt.figure()  # create a figure
plt.gcf()  # get a reference to the active figure
ax1 = fig.add_subplot(2, 2, 1)  # add a subplot to current figure
ax2 = fig.add_subplot(2, 2, 2)  # add a subplot to current figure
ax3 = fig.add_subplot(2, 2, 3)  # add a subplot to current figure
plt.plot(np.random.randn(50).cumsum(), 'k--')  # creates a plot on the last subfigure
ax2.plot(np.random.randn(50).cumsum(), 'b--')  # creates a plot on subfigure n°2
ax1.hist(np.random.randn(100), bins=20, color='r', alpha=0.3)  # creates a histogram on subfigure n°2
fig, axes = plt.subplots(2, 3)  # creates a figure and a NymPy array of subplot objects

# plotting with high level Pandas API
# -----------------------------------
# create a line plot from a Series
s = pd.Series(np.random.randn(100).cumsum(), index=np.arange(0, 100))
s.plot()

# create a line plot for each row in a DataFrame
df = pd.DataFrame(np.random.randn(10, 4).cumsum(0), columns=['A', 'B', 'C', 'D'], index=np.arange(0, 100, 10))
df.plot()

# create two bar plots on two different subplots
fig, axes = plt.subplots(2, 1)
data = pd.Series(np.random.rand(16), index=list('abcdefghijklmnop'))
data.plot(kind='bar', ax=axes[0], color='r', alpha=0.7)
data.plot(kind='barh', ax=axes[1], color='g', alpha=0.7)

# create a vertical stacked bar plot
df = pd.DataFrame(np.random.rand(6, 4), index=['one', 'two', 'three', 'four', 'five', 'six'],
                  columns=pd.Index(['A', 'B', 'C', 'D'], name='Genus'))
df.plot(kind='bar')

# create a horizontal stacked bar plot
df.plot(kind='barh', stacked=True, alpha=0.5)

# create a vertical bar plot that represents fractions
tips = pd.read_csv('/Users/pirminlemberger/PycharmProjects/PythonBook/text files/ch08/tips.csv')
party_counts = pd.crosstab(tips.day, tips.size)
party_pcts = party_counts.div(party_counts.sum(1).astype(float), axis=0)
party_pcts.plot(kind='bar', stacked=True)

# create a histogram
tips['tip_pct'] = tips['tip'] / tips['total_bill']
tips['tip_pct'].hist(bins=50)

# create a histogram with a density plot on top
comp1 = np.random.normal(0, 1, size=200)  # N(0, 1)
comp2 = np.random.normal(10, 2, size=200)  # N(10, 4)
values = pd.Series(np.concatenate([comp1, comp2]))
values.hist(bins=100, alpha=0.3, color='g', normed=True)
values.plot(kind='kde', style='r--')

# create a scatter plot
macro = pd.read_csv('/Users/pirminlemberger/PycharmProjects/PythonBook/text files/ch08/macrodata.csv')
data = macro[['cpi', 'm1', 'tbilrate', 'unemp']]
trans_data = np.log(data).diff().dropna()
plt.scatter(trans_data['m1'], trans_data['unemp'])
plt.title('Changes in log %s vs. log %s' % ('m1', 'unemp'))

plt.scatter_matrix(trans_data, diagonal='kde', color='k', alpha=0.3)

# make a scatter plot for each pair of variables in a DataFrame to show their correlations
pd.scatter_matrix(trans_data, diagonal='kde', color='k', alpha=0.3)

# ===============================================
# chapter 9 Data Aggregation and Group Operations
# ===============================================

# apply an aggregation function to a DataFrame
df = pd.DataFrame({'key1': ['a', 'a', 'b', 'b', 'a'],
                   'key2': ['one', 'two', 'one', 'two', 'one'],
                   'data1': np.random.randn(5),
                   'data2': np.random.randn(5)})
grouped = df['data1'].groupby(df['key1'])
grouped.mean()

# enumerate the groups in a DataFrame
for name, group in df.groupby('key1'):
    print('group: ' + name)
    print(group)
    print

# ============
# miscellanous
# ============

# deleting an object from the workspace

x = 1
del x


# class creation and instanciation
class Complex:
    def __init__(self, realpart, imagpart):
        self.r = realpart
        self.i = imagpart

    def hello(self):
        return 'hello world'


z = Complex(3.0, -4.5)
z.r, z.i

# finding indices where a condition is true or false in a DataFrame,
# see http://stackoverflow.com/questions/21800169/python-pandas-get-index-of-rows-which-column-matches-certain-value
boolean_selector = df['col'] > 321
df[boolean_selector].index.tolist()
df[~boolean_selector].index.tolist()

# scaling variables
# see http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler
from sklearn import preprocessing

# elementary method
X = np.array([[1., -1., 2.],
              [2., 0., 0.],
              [0., 1., -1.]])
X_scaled = preprocessing.scale(X)

# using the fit - transform API
scaler = preprocessing.StandardScaler().fit(X)
scaler.mean_
scaler.scale_
scaler.transform(X)

# string formatting, see: https://pyformat.info/#number
print('x{} y{}'.format(1, 2))
print('x{} y{}'.format('one', 'two'))
print('x{:d}'.format(42))
print('x{:06.2f}'.format(3.141592653589793))
print('{:%Y-%m-%d %H:%M}'.format(datetime(2001, 2, 3, 4, 5)))

# compute the difference between two lists
temp1 = ['a', 'b', 'c']
temp2 = ['a', 'b', 'd']
list(set(temp1) ^ set(temp2))

# count the number of NaN in a DataFrame
df = pd.DataFrame(np.arange(12).reshape(3, 4))
df.ix[1, 2] = np.NaN
df.ix[2, 3] = np.NaN
pd.Series(np.isnan(np.concatenate(df.values.tolist()))).value_counts()
