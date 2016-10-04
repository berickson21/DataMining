import random
import numpy
import math
from random import shuffle
from scipy.spatial import distance as dist_lib
from scipy import stats
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


def get_linear_classification(instance, xIndex, slope, intercept):  # Part 1

    return (slope * float(instance[xIndex])) + intercept  # predict y-variable based on the x-variable


# table is the table.
# n is the index of the attribute to classify
# instance - trying to classify
# k - size of comparision set
def knn_classifier(table, n, instance, k):  # Step 2
    training_set = numpy.array(random.sample(table, table.shape[0] * 2/3))
    sub_training_set = training_set[:, [0, 1, 4, 5]].astype(float)
    distances = []
    
    instance_subset = map(float, numpy.array([instance[0], instance[1], instance[4], instance[5]]))
    t_cat = numpy.empty(sub_training_set.shape)
    for i in range(4):
        t = sub_training_set[:, [i]].flatten()
        t_cat[:, i], instance_subset[i] = normalize(t, instance_subset[i])

    
    for row in t_cat:
        distances.append(distance(row, instance_subset))
    t_cat1 = numpy.append(t_cat, training_set[:, [0]].astype(float), axis=1)
    t_cat2 = numpy.append(t_cat1, numpy.vstack(distances), axis=1)
    t_cat2.sort(axis=1)

    top_k_rows = t_cat2[:5]
    label = select_class_label(top_k_rows)
    print('=================================================================================')
    print('STEP 2: k=5 nearest neighbor MPG Classifier')
    print('=================================================================================')
    print('instance:' + str(instance))
    print('class:' + str(label) + ' ' + 'actual: ' + str(instance[0]))
    return label


# row is a row
# instance is a row
# indices is list of indexes
# returns normalized distance for the instance to the given row

def distance(row, instance):

    distances = []
    indices = [0, 1, 2, 3]
    for i in indices:
        distances.append((row[i] - instance[i])**2)
    
    return math.sqrt(sum(distances))



def normalize(column, instance):
    maximum = max(column)
    minimum = min(column)
    minmax = (maximum - minimum) * 1.0    
    column_normed = []
    
    for item in column:
        column_normed.append((item - minimum)/minmax)
    instance_normed = (instance - minimum)/minmax
    return  column_normed, instance_normed


def select_class_label(top_k_rows):

    # print(stats.mode(top_k_rows[:, 5][0]))
    mode = stats.mode(top_k_rows[:, 5] [0])
    return mode[0]
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


def construct_confusion_matrix_knn(small_partition, large_partition, xIndex, yIndex, k):

    init = [[0] * 10] * 10
    confusion = numpy.array(init)
    total = 0

    for row in small_partition:

        c = classification_map(knn_classifier(large_partition, [0, 1, 2], row, k))
        r = classification_map(row[0]) - 1
        confusion[r][c] += 1
        total += 1

    return numpy.matrix(confusion).tolist()


def construct_confusion_matrix(small_partition, large_partition, xIndex, yIndex, k):

    init = [[0] * 10] * 10
    confusion = numpy.array(init)
    total = 0

    reg = regression_line(large_partition, xIndex, yIndex)  # get Linear Regression

    for row in small_partition:

        c = classification_map(get_linear_classification(row, xIndex, reg[0], reg[1]))
        r = classification_map(row[0]) - 1
        confusion[r][c] += 1
        total += 1

    return numpy.matrix(confusion).tolist()


def stratified_k_folds_knn(table, xIndex, yIndex, k):  # Step 3

    partition_len = len(table)/k
    partitions = [table[i:i + partition_len] for i in range(0, len(table), partition_len)]

    init = [[0] * 10] * 10
    confusion = numpy.matrix(init)

    for part in partitions:
        temp = []
        for p in partitions:
            if part is not p:
                temp += p
        confusion += construct_confusion_matrix_knn(part, temp, 6, 0, k)
    matrix = numpy.squeeze(numpy.asarray(confusion))

    return matrix.tolist()


def stratified_k_folds(table, xIndex, yIndex, k):  # Step 3

    partition_len = len(table)/k
    partitions = [table[i:i + partition_len] for i in range(0, len(table), partition_len)]

    init = [[0] * 10] * 10
    confusion = numpy.matrix(init)

    for part in partitions:
        temp = []
        for p in partitions:
            if part is not p:
                temp += p
        confusion += construct_confusion_matrix(part, temp, 6, 0, k)
    matrix = numpy.squeeze(numpy.asarray(confusion))

    return matrix.tolist()


def get_accuracy_of_confusion(matrix):

    matrix = matrix
    total = (sum([sum(row) for row in matrix]))
    accuracies = []

    for i in range(len(matrix)):
        row = matrix[i][:]
        col = [r[i] for r in matrix]
        row.pop(i)
        col.pop(i)
        accuracies.append((total-(sum(col)+sum(row)))/float(total))

    return round(sum(accuracies)/float(len(accuracies)),2)


def predictive_accuracy(table, xIndex, yIndex, k):  # Step 3

    lrg_accuracy_rs = holdout_partition(table, xIndex, yIndex, k)
    lrg_accuracy_st = get_accuracy_of_confusion(stratified_k_folds(table, xIndex, yIndex, k))

    print_double_line('STEP 3: Predictive Accuracy')
    print '\n\tRandomSubsample(k=10, 2:1 Train / Test)'
    print '\t\tLinear Regression: accuracy = ' + str(lrg_accuracy_rs) + ', error rate = ' + str(1 - lrg_accuracy_rs)
    print '\t\tk Nearest Neighbors: accuracy = 0.??, error rate = 0.??'
    print '\tStratified 10-Fold Cross Validation'
    print '\t\tLinear Regression: accuracy = ' + str(lrg_accuracy_st) + ', error rate = ' + str(1 - lrg_accuracy_st)
    print '\t\tk Nearest Neighbors: accuracy = 0.??, error rate = 0.??'


def holdout_partition(table, xIndex, yIndex, k):

    rand = table[:]  # copy table
    part = (len(rand) * 2) / 3  # find partition
    accuracies = []
    for i in range(k/2):

        shuffle(rand)  # shuffle table
        matrix = construct_confusion_matrix(rand[0: part], rand[part:], xIndex, yIndex, k)
        accuracies.append(get_accuracy_of_confusion(matrix))

    return sum(accuracies) / float(len(accuracies))


def print_double_line(string):
    print '=========================================== \n' + string + '\n=========================================== '


def main():

    table = numpy.array(remove_incomplete_rows(read_csv('auto-data.txt')))
    table1 = remove_incomplete_rows(read_csv('auto-data.txt'))

    # linear_regression_classification(table, 6, 0, 5)  # Step 1

    knn_classifier(table, 0, random.choice(table), len(table[0]) * 2/3)
    # predictive_accuracy(table, 6, 0, 10)                    # Step 3

    # stratified_k_folds(table, 6, 0, 10)



if __name__ == '__main__':
    main()