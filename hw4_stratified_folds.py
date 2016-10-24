from hw2 import read_csv, remove_incomplete_rows, get_column
from hw3 import print_confusion
from hw4_Naive_Bayes import NaiveBayes,ContinuousNaiveBayes
from copy import deepcopy
from random import shuffle

import numpy as numpy


class StratifiedFolds:

    def __init__(self, table, indexes, label_index):
        self.table = deepcopy(table)

        self.indexes = indexes
        self.label_index = label_index

        self.labels = []

        for row in self.table:
            label = self.convert(row[self.label_index], [13, 14, 16, 19, 23, 26, 30, 36, 44])
            if label not in self.labels:
                self.labels.append(label)

        self.num_labels = len(self.labels)

    def stratified_k_folds(self, k):

        new_table = deepcopy(self.table)
        shuffle(new_table)

        partition_len = len(new_table)/(k-1)
        partitions = [new_table[i:i + partition_len] for i in range(0, len(new_table), partition_len)]

        init = [[0] * self.num_labels] * self.num_labels
        confusion = numpy.matrix(init)

        for part in partitions:
            temp = []
            for p in partitions:
                if part is not p:
                    temp += deepcopy(p)
            confusion += self.construct_confusion_matrix(part, temp)
        matrix = numpy.squeeze(numpy.asarray(confusion))

        return matrix.tolist()

    def construct_confusion_matrix(self, test_set, training_set):

        classifier = self.classification(training_set)

        init = [[0] * self.num_labels] * self.num_labels
        confusion = numpy.array(init)
        total = 0

        for instance in test_set:

            c = classifier.classify(instance)
            r = classifier.convert(instance[0], [13, 14, 16, 19, 23, 26, 30, 36, 44])

            confusion[r-1][c-1] += 1
            total += 1

        return numpy.matrix(confusion).tolist()

    @staticmethod
    def get_accuracy_of_confusion(matrix):

        total = (sum([sum(row) for row in matrix]))
        accuracies = []

        for i in range(len(matrix)):
            row = matrix[i]
            col = [r[i] for r in matrix]

            accuracies.append((total - (sum(col) + sum(row) - (2*row[i]))) / float(total))

        return round(sum(accuracies) / float(len(accuracies)), 2), accuracies

    def classification(self, training_set):
        return NaiveBayes(training_set, self.indexes, self.label_index)

    def categorize_table(self):

        for row in self.table:
            self.categorize_instance(row)

    def categorize_instance(self, row):

        row[0] = self.convert(row[0], [13, 14, 16, 19, 23, 26, 30, 36, 44])

    @staticmethod
    def convert(value, cutoffs):

        for i, item in enumerate(cutoffs):
            if float(value) < item:
                return i + 1
            elif float(value) > cutoffs[-1]:
                return len(cutoffs) + 1


class ContinuousStratifiedFolds(StratifiedFolds):

    def __init__(self, table, cat_indexes, cont_indexes, label_index):
        StratifiedFolds.__init__(self, table, cat_indexes, label_index)
        self.cont_indexes = cont_indexes

    def classification(self, training_set):
        return ContinuousNaiveBayes(training_set, self.indexes, self.cont_indexes, self.label_index)


def main():

    table = remove_incomplete_rows(read_csv('auto-data.txt'))
    s = StratifiedFolds(table)
    matrix = s.stratified_k_folds(10)
    print s.get_accuracy_of_confusion(matrix)[0]
    for row in matrix:
        print row
    print_confusion(matrix)

