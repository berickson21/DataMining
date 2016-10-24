from copy import deepcopy
from math import sqrt
from random import sample

from scipy import stats

from hw1 import get_column
from hw2 import read_csv, remove_incomplete_rows
from hw3 import classification_map


class KnnClassifier:

    def __init__(self, training_set, indexes, label_index, k):
        self.table = training_set
        self.training_set = deepcopy(training_set)
        self.indexes = indexes
        self.label_index = 3
        self.k = k

    def knn_classifier(self, instance):

        distance = []
        for row in self.training_set:
            distance.append([self.dist(instance, row), row[self.label_index]])

        distance.sort(key=lambda x: x[0])
        top_k_neighbors = distance[0:self.k]

        return self.get_label(top_k_neighbors)

    def dist(self, instance, row):

        accumulator = 0
        # print('Instance: ' + str(instance))
        # print('Row: ' + str(row))
        # print('Indexes: ' + str(self.indexes))
        for i in self.indexes:
            if instance[i] == row[i]:
                accumulator += 1

        return accumulator

    @staticmethod
    def get_label(top_k_neighbors):
        return stats.mode([top[1] for top in top_k_neighbors])[0][0]

    def convert(self, val):

        if val == 'yes':
            return 0
        else:
            return 1

#             if float(value) < item:
#                 return i + 1
#             elif float(value) > cutoffs[-1]:
#                 return len(cutoffs) + 1
