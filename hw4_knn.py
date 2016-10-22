from copy import deepcopy
from math import sqrt
from random import sample

from scipy import stats

from hw1 import get_column_as_floats
from hw2 import read_csv, remove_incomplete_rows
from hw3 import classification_map


class KnnClassifier:

    def __init__(self, training_set, indexes, label_index, k):
        self.table = training_set
        self.training_set = deepcopy(training_set)
        self.indexes = indexes
        self.label_index = label_index
        self.k = k

        # self.normalize_table()

    def knn_classifier(self, instance):

        inst = self.normalize_instance(instance)
        distance = []

        for row in self.training_set:
            distance.append([self.dist(inst, row), row[self.label_index]])

        distance.sort(key=lambda x: x[0])
        top_k_neighbors = distance[0:self.k]

        return self.get_label(top_k_neighbors)

    def dist(self, instance, row):

        return round(sqrt(sum([((float(instance[i]) - float(row[i]))**2) for i in self.indexes])), 3)

    @staticmethod
    def get_label(top_k_neighbors):
        return stats.mode([top[1] for top in top_k_neighbors])[0][0]

    def normalize_table(self):

        for index in self.indexes:
            self.normalize_column(index)

    def normalize_column(self, index):

        column = get_column_as_floats(self.training_set, index)
        maximum = max(column)
        minimum = min(column)
        spread = maximum - minimum

        for row in self.training_set:
            row[index] = round(
                (float(row[index]) - minimum) / float(spread), 3)

    def normalize_instance(self, instance):

        new_instance = instance[:]

        for index in self.indexes:
            column = get_column_as_floats(self.table, index)
            maximum = max(column)
            minimum = min(column)
            spread = maximum - minimum
            new_instance[index] = round(
                (float(new_instance[index]) - minimum) / float(spread), 3)

        return new_instance


def main():

     table = remove_incomplete_rows(read_csv('auto-data.txt'))

     k = KnnClassifier(table, [1, 4, 5], 0, 5)

     for instance in sample(table, 5):
         print '\tinstance: ' + str(instance)
         print '\tclass: ' + str(classification_map(k.knn_classifier(instance))) \
               + ' actual: ' + str(classification_map(instance[0]))


 main()
