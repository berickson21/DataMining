import matplotlib

matplotlib.use('pdf')

import matplotlib.pyplot as pyplot
import numpy as numpy

from hw1 import read_csv, get_column, get_column_as_floats, remove_incomplete_rows

from scipy import stats as stats


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

    pyplot.savefig('scatterplot_'+COLUMN_NAMES[xIndex]+'.pdf')  # save graph


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

    #xrng = numpy.arange(len(column))
    #pyplot.xticks(xrng, column)

    pyplot.gca().get_yaxis().set_visible(False)
    # pyplot.plot(column, y, marker='.', markersize=50, alpha=0.2)
    pyplot.scatter(column, y, marker='.', alpha=0.2, s=5000, color='b')


    pyplot.savefig('step_3_' + COLUMN_NAMES[index] + '.pdf')


def box_plot(table, xIndex, yIndex):



    pyplot.figure()

    data = group_by(table, xIndex)
    xrng = numpy.arange(len(data[0]), 1)

    pyplot.xticks(xrng, data[0])

    pyplot.xlabel(COLUMN_NAMES[xIndex])
    pyplot.ylabel(COLUMN_NAMES[yIndex])
    pyplot.boxplot(data[1])

    pyplot.xlabel(COLUMN_NAMES[6])
    pyplot.ylabel('Count')
    pyplot.boxplot(data[1])

    pyplot.savefig('box_plot.pdf')



def frequency_chart(table, index):

    pyplot.figure()

    freq = frequency(table, index)


# def frequency_chart(freq, index):
#     xs = freq[0]
#     ys = freq[1]

#     xrng = numpy.arange(len(xs))
#     yrng = numpy.arange(max(ys))

#     pyplot.bar(xrng, ys, 0.5, alpha=0.75, align='center', color='r')

#     pyplot.xticks(xrng, freq[0])
#     pyplot.yticks(yrng)
#     pyplot.xlabel(COLUMN_NAMES[index])
#     pyplot.ylabel(COLUMN_NAMES[index])
#     pyplot.grid(True)

#     pyplot.savefig('frequency_chart.pdf')

    xs = freq[0]
    ys = freq[1]


#     xrng = numpy.arange(len(xs))
#     yrng = numpy.arange(max(ys))


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


def create_histogram(table, index, xLabel, yLabel):
    column = get_column(table, index)
    column.sort()


def historgram_continuous(table, index):
    column = get_column_as_floats(table, index)
    column.sort()

    pyplot.figure()

    pyplot.hist(column, bins=10)
    pyplot.xlabel(COLUMN_NAMES[index])  # x label
    pyplot.ylabel('Frequency')  # y label

    pyplot.savefig('histogram_'+COLUMN_NAMES[index]+'.pdf')  # save graph

    pyplot.xlabel(COLUMN_NAMES[index])  # x label
    pyplot.ylabel('Frequency')  # y label

    pyplot.hist(cut_off_frequency(table, index, cutoffs), bins=10, label='EPA MPG Categories')
    pyplot.savefig('histogram_'+COLUMN_NAMES[index]+'.pdf')  # save graph



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



def group(table, index):

    dict = {}

    for row in table:
        if row[index] != 'NA':
            if row[index] in dict:
                dict[row[index]].append(row)
            else:
                dict.update({row[index]: [row]})

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

# def frequency_bins(table, index, cutoffs):

#     freq = [0]*len(cutoffs)
#     cutoffs.sort()
#     col = get_column_as_floats(table, index)


#     for item in col:
#         for i in range(len(cutoffs)):
#             if item <= cutoffs[i]:
#                 freq[i] += 1
#                 break
#             elif item > max(cutoffs):
#                 freq[-1] += 1
#     return  freq


def cut_off_frequency(table, index, cutoffs): #Step 4.1

    freq = [0]*(len(cutoffs))

    col = get_column_as_floats(table, index)

    for item in col:
        for i in range(len(cutoffs)):
            if item <= cutoffs[i]:
                freq[i] += 1
                break
    return  freq, cutoffs

def regression_line(table, index_x, index_y):
    length = max(len(table[index_x]), len(table[index_y]))
    list_x = get_column_as_floats(table, index_x)
    list_y = get_column_as_floats(table, index_y)
    print(len(list_x))
    print(len(list_y))
    return stats.linregress(list_x, list_y)


def get_regression_lines(table):
    r_line_disp = regression_line(table, 2, 0)
    r_line_horses = regression_line(table, 3, 0)
    r_line_weight = regression_line(table, 4, 0)
    r_line_msrp = regression_line(table, 9, 0)
    pyplot.figure()


    return freq


def transform_frequency_chart(table, index, cutoffs, part):

    freq = cut_off_frequency(table, 0, cutoffs)
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


def divided_frequency_chart(table, index1, index2):

    sub1 = group(table, index2)
    dict = {}

    for sub in sub1:
        g = group(sub, index1)
        key = sub[0][index2]
        freq = []
        for item in g:
            freq.append(len(item))
        dict.update({key: freq})

    dict = sort_dict(dict)
    xLables = dict[0]
    values = [i for i in range(70, 80, 1)]
    pyplot.figure()

    index = numpy.arange(10)

    pyplot.bar(index, xLables[0], width=.3, alpha=.5, color='lightblue', label='US')
    pyplot.bar(index+.3, xLables[1], width=.3, alpha=.5, color='red', label='Europe')
    pyplot.bar(index+.6, xLables[2], width=.3, alpha=.5, color='gold', label='Japan')

    pyplot.xlabel('Model Year')
    pyplot.ylabel('Count')
    pyplot.title('Total number of cars by Model Year and Country of Origin')
    pyplot.xticks(index + 0.5, values)
    pyplot.legend()

    pyplot.savefig('step_8_partB.pdf')


    pyplot.scatter(4, 0)
    pyplot.plot(r_line_weight)
    pyplot.xlabel('Weight')
    pyplot.ylabel('MPG')
    pyplot.savefig('scatterplot_lin_reg.pdf')

def create_scatterplots(table):
    scatter_plot(table, 2, 0)
    scatter_plot(table, 3, 0)
    scatter_plot(table, 4, 0)
    scatter_plot(table, 5, 0)
    scatter_plot(table, 9, 0)

# def create_histograms(table):
#     frequency_chart(table, 1)
#     frequency_chart(table, 6)
#     frequency_chart(table, 7)

def create_histograms_continuous(table):
    historgram_continuous(table, 0)
    historgram_continuous(table, 2)
    historgram_continuous(table, 3)
    historgram_continuous(table, 4)
    historgram_continuous(table, 5)
    historgram_continuous(table, 9)


def main():

    table = read_csv('auto-data.txt')

    freq = cut_off_frequency(table, 0, get_cutoffs(table, 0, 10))

    # Step 1
    frequency_chart(table, 1)
    frequency_chart(table, 6)
    frequency_chart(table, 7)

    # Step 2
    pie_chart(table, 1)
    pie_chart(table, 6)
    pie_chart(table, 7)

    # Step 3
    strip_char(table, 0)
    strip_char(table, 2)
    strip_char(table, 4)
    strip_char(table, 5)
    strip_char(table, 9)

    # Step 4 Part A
    cuts = [13, 14, 16, 19, 23, 26, 30, 36, 44]
    transform_frequency_chart(table, 0, cuts, 'A')

    # Step 4 Part B
    cuts = get_cutoffs(table, 0, 5)
    transform_frequency_chart(table, 0, cuts, 'B')

    # Step 5

    # Step 6

    # Step 7

    freq = cut_off_frequency(table, 0, get_cutoffs(table, 0, 10))

    # create_histograms(table)            #Step 1
    pie_char(freq)                      #Step 2
    strip_char(table, 0)                #Step 3
    create_histograms_continuous(table) #Step 5
    create_scatterplots(table)          #Step 6
    # get_regression_lines(table)         #Step 7
    box_plot(table, 6)                  #Step 8


    # Step 8
    box_plot(table, 6, 0)
    divided_frequency_chart(table, 6, 7)

main()
