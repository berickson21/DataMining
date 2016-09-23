import matplotlib

matplotlib.use('pdf')

import matplotlib.pyplot as pyplot
import numpy as numpy

from hw1 import read_csv, get_column, get_column_as_floats

COLUMN_NAMES = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Orgin', 'Car Name', 'MSRP']


def scatter_plot(table, xIndex, yIndex, xLabel, yLabel):

    ys = []
    xs = []

    for row in table:  # get x and y values from table
        if row[xIndex] != 'NA' and row[yIndex] != 'NA':
            ys.append(float(row[yIndex]))
            xs.append(float(row[xIndex]))

    pyplot.figure()

    pyplot.xlim(int(min(xs)) * 0.95, int(max(xs) * 1.05))  # set x bounds on graph
    pyplot.ylim(int(min(ys)) * 0.95, int(max(ys) * 1.05))  # set y bounds on graph

    pyplot.xlabel(xLabel)  # x label
    pyplot.ylabel(yLabel)  # y label

    pyplot.grid()

    pyplot.scatter(xs, ys, color='g')

    pyplot.savefig('fig8.pdf')  # save graph


def pie_chart(table, index):

    freq = frequency(table, index)

    total = sum(freq[1])  # get total (probably should be length)
    percents = []

    for item in freq[1]:
        percents.append(item / float(total))

    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']

    pyplot.figure()
    pyplot.title('Total Number of cars by ' + COLUMN_NAMES[index])
    pyplot.pie(percents, labels=freq[0], autopct='%1.1f%%',  colors=colors)

    pyplot.savefig('step_2_' + COLUMN_NAMES[index] + '.pdf')


def strip_char(table, index):

    column = get_column_as_floats(table, index)
    y = [1] * len(column)

    pyplot.figure()

    pyplot.title(COLUMN_NAMES[index] + ' of all Cars')
    pyplot.xlabel(COLUMN_NAMES[index])

    xrng = numpy.arange(len(column))
    pyplot.xticks(xrng, column)

    pyplot.gca().get_yaxis().set_visible(False)
    pyplot.plot(column, y, marker='.', markersize=50, alpha=0.2)

    pyplot.savefig('step_3_' + COLUMN_NAMES[index] + '.pdf')


def box_plot(table, index, xLabel, yLabel):

    pyplot.figure()

    data = group_by(table, index)
    xrng = numpy.arange(len(data[0]), 1)

    pyplot.xticks(xrng, data[0])
    pyplot.xlabel(xLabel)
    pyplot.ylabel(yLabel)
    pyplot.boxplot(data[1])

    pyplot.savefig('fig7.pdf')


def frequency_chart(freq, index):

    pyplot.figure()

    xs = freq[0]
    ys = freq[1]

    xrng = numpy.arange(len(xs))
    yrng = numpy.arange(max(ys) + 2)

    pyplot.bar(xrng, ys, 0.5, alpha=0.75, align='center', color='r')

    pyplot.xticks(xrng, freq[0])

    pyplot.title('Number of Cars by ' + COLUMN_NAMES[index])
    pyplot.xlabel(COLUMN_NAMES[index])
    pyplot.ylabel('Count')

    pyplot.grid(False)

    pyplot.savefig('step_1_' + COLUMN_NAMES[index] + '.pdf')


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


def create_histogram(table, index, xLabel, yLabel):
    column = get_column(table, index)
    column.sort()

    cutoffs = [13, 14, 16, 19, 23, 26, 30, 36, 44, 45]

    pyplot.figure()

    pyplot.hist(cut_off_frequency(table, index, cutoffs), bins=10, label='EPA MPG Categories')

    pyplot.savefig('fig8.pdf')

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
        for i in range(len(cutoffs)):
            if item <= cutoffs[i]:
                freq[i] += 1
                break

    return cutoffs, freq


def main():

    table = read_csv('auto-data.txt')

    frequency_chart(frequency(table, 1), 1)
    frequency_chart(frequency(table, 6), 6)
    frequency_chart(frequency(table, 7), 7)

    pie_chart(table, 1)
    pie_chart(table, 6)
    pie_chart(table, 7)

    strip_char(table, 0)
    strip_char(table, 2)
    strip_char(table, 4)
    strip_char(table, 5)
    strip_char(table, 9)

    cuts = [13, 14, 16, 19, 23, 26, 30, 36, 44]
    freq =

    strip_char(table, 0)
    box_plot(table, 6, 'Year', 'MPG')
    scatter_plot(table, 6, 0, 'Year', 'MPG')


main()