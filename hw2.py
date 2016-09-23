import matplotlib

matplotlib.use('pdf')

import matplotlib.pyplot as pyplot
import numpy as numpy

from hw1 import read_csv, get_column, get_column_as_floats

COLUMN_NAMES = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Orgin', 'Car Name', 'MSRP']


def scatter_plot(table, xIndex, yIndex):

    ys = []
    xs = []

    for row in table:  # get x and y values from table
        if row[xIndex] != 'NA' and row[yIndex] != 'NA':
            ys.append(float(row[yIndex]))
            xs.append(float(row[xIndex]))

    pyplot.figure()

    pyplot.xlim(int(min(xs)) * 0.95, int(max(xs) * 1.05))  # set x bounds on graph
    pyplot.ylim(int(min(ys)) * 0.95, int(max(ys) * 1.05))  # set y bounds on graph

    pyplot.xlabel('Year')  # x label
    pyplot.ylabel('MPG')  # y label

    pyplot.grid()

    pyplot.scatter(xs, ys, color='g')

    pyplot.savefig('fig8.pdf')  # save graph


def pie_char(freq):

    total = sum(freq[1])  # get total (probably should be length)
    percents = []

    for item in freq[1]:
        percents.append(item / float(total))

    pyplot.figure()
    pyplot.pie(percents, labels=freq[0])

    pyplot.savefig('fig3.pdf')


def strip_char(table, index):

    column = get_column_as_floats(table, index)
    y = [1] * len(column)

    pyplot.figure()

    xrng = numpy.arange(len(column))
    pyplot.xticks(xrng, column)

    pyplot.gca().get_yaxis().set_visible(False)
    pyplot.plot(column, y, marker='.', markersize=50, alpha=0.2)

    pyplot.savefig('fig4.pdf')


def box_plot(table, index):

    pyplot.figure()

    data = group_by(table, index)
    xrng = numpy.arange(len(data[0]), 1)

    pyplot.xticks(xrng, data[0])
    pyplot.xlabel('Year')
    pyplot.ylabel('MPG')
    pyplot.boxplot(data[1])

    pyplot.savefig('fig7.pdf')


def frequency_chart(freq):
    xs = freq[0]
    ys = freq[1]

    xrng = numpy.arange(len(xs))
    yrng = numpy.arange(max(ys) + 2)

    pyplot.bar(xrng, ys, 0.5, alpha=0.75, align='center', color='r')

    pyplot.xticks(xrng, freq[0])
    pyplot.yticks(yrng)

    pyplot.grid(True)

    pyplot.savefig('fig2.pdf')


def frequency(table, index):
    # returns frequency of values in a column

    column = get_column(table, index)
    column.sort()

    cats = []

    for item in column:
        if item not in cats:
            cats.append(item)

    freq = [0]*len(cats)

    for item in column:
        for index in range(len(cats)):
            if item == cats[index]:
                freq[index] += 1

    return cats, freq


def group_by(table, index):

    dict = {}

    for row in table:
        if row[0] != 'NA':
            if row[index] in dict:
                dict[row[index]].append(float(row[0]))
            else:
                dict.update({row[index]: [float(row[0])]})

    keys = sorted(dict.keys())
    values = []

    for key in keys:
        values.append(dict[key])

    return keys, values


def get_cutoffs(table, index, num):

    col = get_column_as_floats(table, index)

    max_value = int(max(col))
    min_value = int(min(col))

    width = int((max_value-min_value)/num)

    return list(range(min_value + width, max_value+1, width))


def cut_off_frequency(table, index, cutoffs):

    freq = [0]*len(cutoffs)

    cutoffs.sort()
    col = get_column_as_floats(table, index)

    for item in col:
        for index in range(len(cutoffs)):
            if item <= cutoffs[index]:
                freq[index] += 1
                break

    return cutoffs, freq


def main():

    table = read_csv('auto-data.txt')

    frequency_chart(frequency(table, 0))
    freq = cut_off_frequency(table, 0, get_cutoffs(table, 0, 10))
    pie_char(freq)

    strip_char(table, 0)
    box_plot(table, 6)
    scatter_plot(table, 6, 0)


main()