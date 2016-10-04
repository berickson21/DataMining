import random
import numpy
from scipy.spatial import distance as dist_lib

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

def distance(row, instance, indices):
    comp_row = row[indices]
    comp_instance = instance[indices]
    distances = dist_lib.euclidean(comp_row, comp_instance)
    print (numpy.linalg.norm(distances))
    return (numpy.linalg.norm(distances))


def normalize(col):

    maximum = max(col)
    minimum = min(col)
    rng = maximum - minimum
    return[(item - minimum)/float(rng) for item in col]


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

    knn_classifier(table, 0, random.choice(table), len(table[0]) * 2/3)
    # linear_regression_classification(table, 6, 0, 5)  # Step 1

if __name__ == '__main__':
    main()