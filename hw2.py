import matplotlib

matplotlib.use('pdf')

import matplotlib.pyplot as pyplot
import numpy as numpy

from hw1 import read_csv, get_column, get_column_as_floats


from scipy import stats as stats


COLUMN_NAMES = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin', 'Car Name', 'MSRP']


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

    pyplot.xlabel(COLUMN_NAMES[xIndex])  # x label
    pyplot.ylabel(COLUMN_NAMES[yIndex])  # y label

    pyplot.grid()

    pyplot.scatter(xs, ys, color='g')

    pyplot.savefig('step_6_'+COLUMN_NAMES[xIndex]+'.pdf')  # save graph


def pie_chart(table, index):

    freq = frequency(table, index)

    total = sum(freq[1])  # get total (probably should be length)
    percents = []

    for item in freq[1]:
        percents.append(item / float(total))

    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'lightblue']

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

    pyplot.gca().get_yaxis().set_visible(False)
    pyplot.scatter(column, y, marker='.', alpha=0.2, s=5000, color='b')

    pyplot.savefig('step_3_' + COLUMN_NAMES[index] + '.pdf')


def box_plot(table, xIndex, yIndex):

    pyplot.figure()

    data = group_by(table, xIndex)
    xrng = numpy.arange(len(data[0]), 1)

    pyplot.xticks(xrng, data[0])

    pyplot.xlabel(COLUMN_NAMES[xIndex])
    pyplot.ylabel(COLUMN_NAMES[yIndex])
    pyplot.title(COLUMN_NAMES[yIndex] + ' by ' + COLUMN_NAMES[xIndex])
    pyplot.boxplot(data[1])

    pyplot.xlabel(COLUMN_NAMES[xIndex])
    pyplot.ylabel(COLUMN_NAMES[yIndex])
    pyplot.boxplot(data[1])

    pyplot.savefig('step_8_partA.pdf')


def frequency_chart(table, index):

    pyplot.figure()

    freq = frequency(table, index)

    xs = freq[0]
    ys = freq[1]

    xrng = numpy.arange(len(xs))
    yrng = numpy.arange(max(ys))

    pyplot.bar(xrng, ys, 0.5, alpha=0.75, align='center', color='lightblue')

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


def historgram_continuous(table, index):

    column = get_column_as_floats(table, index)
    column.sort()

    pyplot.figure()

    pyplot.xlabel(COLUMN_NAMES[index])  # x label
    pyplot.ylabel('Frequency')  # y label

    pyplot.hist(column, bins=10, label='EPA MPG Categories')
    pyplot.savefig('step_5_'+COLUMN_NAMES[index]+'.pdf')  # save graph


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



def group(table, index, keys):

    dict = {key: [] for key in keys}

    for row in table:
        dict[row[index]].append(row)

    return sort_dict(dict)[0]


def sort_dict(dictionary):

    keys = sorted(dictionary.keys())
    values = []

    for key in keys:
        values.append(dictionary[key])

    return values, keys


def get_cutoffs(table, index, num):     #Step 4.2

    col = get_column_as_floats(table, index)

    max_value = int(max(col))
    min_value = int(min(col))

    width = int((max_value-min_value)/num)

    return list(range(min_value + width, max_value+1, width))


def cut_off_frequency(table, index, cutoffs): #Step 4.1

    freq = [0]*(len(cutoffs))

    col = get_column_as_floats(table, index)

    for item in col:
        for i in range(len(cutoffs)):
            if item <= cutoffs[i]:
                freq[i] += 1
                break
    return freq, cutoffs


def regression_line(table, index_x, index_y):
    list_x = get_column_as_floats(table, index_x)
    list_y = get_column_as_floats(table, index_y)
    return stats.linregress(list_x, list_y)


def get_regression_lines(table):

    r_line_disp = regression_line(table, 2, 0)
    r_line_horses = regression_line(table, 3, 0)
    r_line_weight = regression_line(table, 4, 0)
    r_line_msrp = regression_line(table, 9, 0)

    slope, intercept, r_value, p_value, slope_std_error = stats.linregress(numpy.asarray(get_column_as_floats(table, 4)), numpy.asarray(get_column_as_floats(table, 0)))
    xs = get_column_as_floats(table, 4)
    ys = get_column_as_floats(table, 0)
    pyplot.figure()

    pyplot.scatter(xs, ys)
    pyplot.plot ([slope * x + intercept for x in range(0, int(max(xs)))], color='r')
    pyplot.savefig('step_7_Weight.pdf')
    


def transform_frequency_chart(table, index, cutoffs, part):

    freq = cut_off_frequency(table, 0, cutoffs)[0]
    # xLabels = [i + 1 for i in range(len(freq))]

    labels = make_labels_from_cutoffs(cutoffs)
    pyplot.figure()


    xrng = numpy.arange(len(freq))

    pyplot.xticks(xrng, labels)
    pyplot.ylabel('Count')
    pyplot.xlabel(COLUMN_NAMES[index])

    if part == 'A':
        pyplot.title('Total Number of Cars by Equal Width Rankings of ' + COLUMN_NAMES[index])
    else:
        pyplot.title('Total Number of Cars by Equal Width Rankings of ' + COLUMN_NAMES[index])

    pyplot.bar(xrng, freq, alpha=.75, width=0.5, align='center', color='r')
    pyplot.savefig('step_4_' + COLUMN_NAMES[index] + part + '.pdf')


def make_labels_from_cutoffs(cutoffs):

    labels = []

    for index in range(len(cutoffs)):
        if index == 0:
            labels.append('$\leq$' + str(cutoffs[index]))
        elif index == len(cutoffs)-1:
            labels.append('$\geq$' + str(cutoffs[index]))
        else:
            labels.append(str(cutoffs[index]) + '-' + str(cutoffs[index + 1] - 1))

    return labels


def get_column_nodups(table, index):

    col = []

    for row in table:
        if row[index] not in col:
            col.append(row[index])
    col.sort()
    return col


def count_if(table, index, key):

    count = 0

    for row in table:
        if row[index] == key:
            count += 1

    return count


def divided_frequency_chart(table, index1, index2):

    col1 = get_column_nodups(table, index1)
    col2 = get_column_nodups(table, index2)

    sub1 = group(table, index2, col2)

    l = len(col1)
    values = [[0]*l, [0]*l, [0]*l]

    for i in range(len(sub1)):
        for j in range(len(col1)):
            values[i][j] = count_if(sub1[i], index1, col1[j])

    pyplot.figure()

    index = numpy.arange(10)
    xLables[1].append(0) # FIX


    pyplot.bar(index, values[0], width=.3, alpha=.5, color='lightblue', label='US')
    pyplot.bar(index+.3, values[1], width=.3, alpha=.5, color='red', label='Europe')
    pyplot.bar(index+.6, values[2], width=.3, alpha=.5, color='gold', label='Japan')

    pyplot.xlabel('Model Year')
    pyplot.ylabel('Count')
    pyplot.title('Total number of cars by Model Year and Country of Origin')
    pyplot.xticks(index + 0.5, col1)
    pyplot.legend(loc=2)

    pyplot.savefig('step_8_partB.pdf')


def remove_incomplete_rows(table):

    newTable = []

    for row in table:
        for item in row:
            check = True
            if item == 'NA':
                check = False
                break
        if check:
            newTable.append(row)

    return newTable


def main():

    table = read_csv('auto-data.txt')
    table = remove_incomplete_rows(table)
<<<<<<< HEAD

    print table[0]
    freq = cut_off_frequency(table, 0, get_cutoffs(table, 0, 10))
=======
>>>>>>> refs/remotes/origin/master

    # Step 1
    frequency_chart(table, 1)
    frequency_chart(table, 6)
    frequency_chart(table, 7)
    pyplot.close("all")

    # Step 2
    pie_chart(table, 1)
    pie_chart(table, 6)
    pie_chart(table, 7)
    pyplot.close("all")

    # Step 3
    strip_char(table, 0)
    strip_char(table, 2)
    strip_char(table, 4)
    strip_char(table, 5)
    strip_char(table, 9)
    pyplot.close("all")

    # Step 4 Part A
    cuts = [13, 14, 16, 19, 23, 26, 30, 36, 44]
    transform_frequency_chart(table, 0, cuts, 'A')
    pyplot.close("all")

    # Step 4 Part B
    cuts = get_cutoffs(table, 0, 5)
    transform_frequency_chart(table, 0, cuts, 'B')
    pyplot.close("all")

    # Step 5
    historgram_continuous(table, 0)
    historgram_continuous(table, 2)
    historgram_continuous(table, 3)
    historgram_continuous(table, 4)
    historgram_continuous(table, 5)
    historgram_continuous(table, 9)
<<<<<<< HEAD
=======
    pyplot.close("all")
>>>>>>> refs/remotes/origin/master

    # Step 6
    scatter_plot(table, 2, 0)
    scatter_plot(table, 3, 0)
    scatter_plot(table, 4, 0)
    scatter_plot(table, 5, 0)
    scatter_plot(table, 9, 0)
<<<<<<< HEAD

    # Step 7
    get_regression_lines(table)
=======
    pyplot.close("all")

    # Step 7
    # get_regression_lines(table)
>>>>>>> refs/remotes/origin/master

    # Step 8
    box_plot(table, 6, 0)
    divided_frequency_chart(table, 6, 7)

    pyplot.close("all")

main()
