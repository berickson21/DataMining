import random
import numpy
import math
from scipy.spatial import distance as dist_lib
from operator import sub

from hw1 import read_csv, maximum, get_column_as_floats
from hw2 import remove_incomplete_rows, regression_line, COLUMN_NAMES


# table is an instance of a table
# instance in the row with the missing attribute we are predicting
# xIndex is the column index that corresponds to the predictive value
# yIndex is the column index that corresponds to the value being predicted
# returns predicted value for yIndex attribute of instance

def get_linear_regression_classification(table, instance, xIndex, yIndex):  # Part 1

    reg = regression_line(table, xIndex, yIndex)  # get Linear Regression
    slope = reg[0]   # get slope
    intercept = reg[1]  # get intercept

    return (slope * float(instance[xIndex])) + intercept  # predict y-variable based on the x-variable

# table is the table.
# n is the index of the attribute to classify
# instance - trying to classify
# k - size of comparision set

def knn_classifier(table, n, instance, k):  # Step 2
    training_set = numpy.array(random.sample(table, table.shape[0] * 2/3))
    sub_training_set = training_set[:, [0, 1, 4, 5]]
    sub_training_set = sub_training_set.astype(float)
    distances = []
    # training_set_normed = numpy.empty(sub_training_set.shape)
    sub_training_set[:, [0]] = normalize(sub_training_set[:, [0]])
    sub_training_set[:, [2]] = normalize(sub_training_set[:, [2]])
    sub_training_set[:, [3]] = normalize(sub_training_set[:, [3]])
    # print(sub_training_set)
    # instance_subset = instance[0, 1, 4, 5]
    # print(instance_subset)

    # for row in sub_training_set[[1], :]:
    #     # print("==================================================================================================")
    #     # print('instance is' + str(instance))
    #     # print(row)
    #     new_row = distance(row, instance)
    #     distances.append(new_row)

    # distances.sort()
    # print('non-normalized distances:')
    # print(distances)
    # print('normalized distances:')
    # print(distances_normed)
    top_k_rows = distances[:k]
    label = select_class_label(top_k_rows)

    return label


# row is a row
# instance is a row
# indices is list of indexes
# returns normalized distance for the instance to the given row

def distance(row, instance):
    print(row)
    internal_list = map(numpy.subtract, numpy.asarray(row, dtype=float), numpy.asarray(instance, dtype=float))
    # print('internal distance:' + str(internal_list))
    euclidean_distance = math.sqrt(sum(x**2 for x in internal_list))
    print(euclidean_distance)
    # distances = dist_lib.euclidean(comp_row, comp_instance)

    return euclidean_distance


def normalize(col):
    maximum = numpy.max(col)
    minimum = numpy.min(col)
    minmax = (maximum - minimum) * 1.0
    return[(item - minimum)/minmax for item in col]


def select_class_label(top_k_rows):

    return 'Label'


def linear_regression_classification(table, xIndex, yIndex, k):  # step 1

    print_double_line('STEP 1: Linear Regression MPG Classifier')

    map = [13, 14, 16, 19, 23, 26, 30, 36, 44]


    for instance in random.sample(table, k):
        print '\tinstance: ' + str(instance)
        print '\tclass: ' + str(instance[0], map) + ' actual: ' + str(get_linear_regression_classification(table, instance, xIndex, yIndex))


def classification_map(value, map):

    for index in range(len(map)):
        if value >= map[index]:
            return index + 1



def print_double_line(string):
    print '=========================================== \n' + string + '\n=========================================== '


def main():

    table = numpy.array(remove_incomplete_rows(read_csv('auto-data.txt')))

    knn_classifier(table, 0, random.choice(table[[1], :]), len(table[0]) * 2/3)
    # linear_regression_classification(table, 6, 0, 5)  # Step 1

if __name__ == '__main__':
    main()