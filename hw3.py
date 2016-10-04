import random
<<<<<<< HEAD
import math
import numpy as numpy
from random import shuffle
from hw1 import read_csv
from hw2 import remove_incomplete_rows, regression_line
=======
import numpy
from scipy.spatial import distance as dist_lib

from hw1 import read_csv, maximum, get_column_as_floats
from hw2 import remove_incomplete_rows, regression_line, COLUMN_NAMES
>>>>>>> master


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

<<<<<<< HEAD

def get_linear_classification(instance, xIndex, slope, intercept):  # Part 1

    return (slope * float(instance[xIndex])) + intercept  # predict y-variable based on the x-variable


# training_set is a subset of the table
# n is the number of at attributes
=======
# table is the table.
# n is the index of the attribute to classify
>>>>>>> master
# instance - trying to classify
# k - size of comparision set

def knn_classifier(table, n, instance, k):  # Step 2

    print_double_line('STEP 2: k=' + k + 'Nearest Neighbor MPG Classifier')

    distances = []

    training_set = random.sample(table, len(table[n]) * 2/3)


    for row in training_set:
        distances.append([distance(row, instance, []), row])

    distances.sort(key=lambda x: x[0])
    top_k_rows = distances[:k]
    label = select_class_label(top_k_rows)

    return label


# row is a row
# instance is a row
# indices is list of indexes
# returns normalized distance for the instance to the given row

<<<<<<< HEAD
def distance(row, instance, n):

    return math.sqrt(sum([((row[index]-instance[index]) ** 2) for index in n]))
=======
def distance(row, instance, indices):
    comp_row = row[indices]
    comp_instance = instance[indices]
    distances = dist_lib.euclidean(comp_row, comp_instance)
    print (numpy.linalg.norm(distances))
    return (numpy.linalg.norm(distances))
>>>>>>> master


def normalize(col):

    maximum = max(col)
    minimum = min(col)
    rng = maximum - minimum
    return[(item - minimum)/float(rng) for item in col]


def select_class_label(top_k_rows):

    return 'Label'


def linear_regression_classification(table, xIndex, yIndex, k):  # step 1

    print_double_line('STEP 1: Linear Regression MPG Classifier')

    for instance in random.sample(table, k):
        print '\tinstance: ' + str(instance)
        print '\tclass: ' + str(classification_map(get_linear_regression_classification(table, instance, xIndex, yIndex))) \
              + ' actual: ' + str(classification_map(instance[0]))


def classification_map(value, map=[13, 14, 16, 19, 23, 26, 30, 36, 44]):

    for index in range(len(map)):
        if float(value) <= map[index]:
            return index + 1


def predictive_accuracy(table, xIndex, yIndex):  # Step 3

    init = [[0]*10]*10
    confusion = numpy.array(init)

    partitions = holdout_partition(table)

    reg = regression_line(partitions[0], xIndex, yIndex)  # get Linear Regression
    total = 0

    for row in partitions[1]:

        c = classification_map(get_linear_classification(row, xIndex, reg[0], reg[1]))
        r = classification_map(row[0])-1
        confusion[r][c] += 1
        total += 1

    for row in confusion:
        print row

    # print_double_line('STEP 3: Predictive Accuracy')
    # print '\n\tRandomSubsample(k=10, 2:1 Train / Test)'
    # print '\t\tLinear Regression: accuracy = 0.??, error rate = 0.??'
    # print '\t\tk Nearest Neighbors: accuracy = 0.??, error rate = 0.??'
    # print '\tStratified 10-Fold Cross Validation'
    # print '\t\tLinear Regression: accuracy = 0.??, error rate =  0.??'
    # print '\t\tk Nearest Neighbors: accuracy = 0.??, error rate = 0.??'


def holdout_partition(table):

    rand = table[:]  # copy table
    shuffle(rand)  # shuffle table

    part = (len(rand)*2)/3  # find partition

    return rand[0: part], rand[part:]


def print_double_line(string):
    print '=========================================== \n' + string + '\n=========================================== '


def main():

<<<<<<< HEAD
    table = read_csv('auto-data.txt')
    table = remove_incomplete_rows(table)
    #
    # linear_regression_classification(table, 6, 0, 5)  # Step 1

    predictive_accuracy(table, 6, 0)                        # Step 3

=======
    table = numpy.array(remove_incomplete_rows(read_csv('auto-data.txt')))

    knn_classifier(table, 0, random.choice(table), len(table[0]) * 2/3)
    # linear_regression_classification(table, 6, 0, 5)  # Step 1
>>>>>>> master

if __name__ == '__main__':
    main()