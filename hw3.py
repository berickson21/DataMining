import random
import numpy
import math
from tabulate import tabulate
from random import randint
from random import shuffle
from scipy.spatial import distance as dist_lib
from scipy import stats
from operator import sub
from hw1 import read_csv, maximum, get_column_as_floats
from hw2 import remove_incomplete_rows, knn, regression_line, COLUMN_NAMES




# table is the table.
# n is the index of the attribute to classify
# instance - trying to classify
# k - size of comparision set
def knn_classifier(training_set, n, instance, k):  # Step 2
    sub_training_set = training_set[:, [0, 1, 4, 5]].astype(float)
    distances = []

    instance_subset = map(float, numpy.array([instance[0], instance[1], instance[4], instance[5]]))
    t_cat = numpy.empty(sub_training_set.shape)
    for i in range(1, 3):
        t = sub_training_set[:, [i]].flatten()
        t_cat[:, i], instance_subset[i] = normalize(t, instance_subset[i])

    for row in t_cat:
        distances.append(distance(row, instance_subset))
    t_cat1 = numpy.append(t_cat, numpy.vstack(distances), axis=1)
    t_cat1.sort(axis=1)

    top_k_rows = t_cat1[:4]
    label = select_class_label(top_k_rows)
    
    print_double_line('STEP 2: k=5 nearest neighbor MPG Classifier')
    print('\tinstance:' + str(instance))
    print('\tclass:' + str(classification_map(label)) + ' ' + 'actual: ' + str(classification_map(instance[0])))
    return label


# table is the table.
# n is the index of the attribute to classify
# instance - trying to classify
# k - size of comparision set
def knn_classifier_predictor(training_set, instance):  # Step 2
    training_set = numpy.array(training_set)
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
    return label

# row is a row
# instance is a row
# indices is list of indexes
# returns normalized distance for the instance to the given row
def distance(row, instance):
    distances = []
    indices = [0, 1, 2, 3]
    accumulator=[]
    for i in indices:
        accumulator.append((row[i] + instance[i])**2)
    return math.sqrt(sum(accumulator)) 
    
    


def normalize(column, instance):
    maximum = max(column)
    minimum = min(column)
    minmax = (maximum - minimum) * 1.0    
    column_normed = []
    
    for item in column:
        column_normed.append((item - minimum)/minmax)
    instance_normed = (instance - minimum)/minmax
    return column_normed, instance_normed


def select_class_label(top_k_rows):

    return stats.mode(top_k_rows[:, 0] [0])



def main():

    table = numpy.array(remove_incomplete_rows(read_csv('auto-data.txt')))
    two_arrays = numpy.split(table, [len(table)*2/3])
    training_set = numpy.array(random.sample(two_arrays[0], len(two_arrays[0])))    
    test_set = numpy.array(random.sample(two_arrays[1], len(two_arrays[1])))

    for item in test_set:                           # Step 2
        knn_classifier(table, 0, item, 5)       



if __name__ == '__main__':
    main()
