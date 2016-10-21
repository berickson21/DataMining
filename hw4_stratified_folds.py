from hw2 import read_csv, remove_incomplete_rows
from hw3 import print_confusion
from hw4_Naive_Bayes import NaiveBayes
from copy import deepcopy
from random import shuffle

import numpy as numpy


class StratifiedFolds:

    def __init__(self, table):
        self.table = deepcopy(table)

    def stratified_k_folds(self, k):

        new_table = deepcopy(self.table)
        shuffle(new_table)

        partition_len = len(new_table)/(k-1)
        partitions = [new_table[i:i + partition_len] for i in range(0, len(new_table), partition_len)]

        init = [[0] * 10] * 10
        confusion = numpy.matrix(init)

        for part in partitions:
            temp = []
            for p in partitions:
                if part is not p:
                    temp += p
            confusion += self.construct_confusion_matrix(part, temp, 5, 10)
        matrix = numpy.squeeze(numpy.asarray(confusion))

        return matrix.tolist()

    def construct_confusion_matrix(self, test_set, training_set, k, num_labels):

        indexes = [1, 4, 6]

        classifier = self.classification(training_set, indexes, 0)

        init = [[0] * num_labels] * num_labels
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

            accuracies.append((total - (sum(col) + sum(row))) / float(total))

        return round(sum(accuracies) / float(len(accuracies)), 2), accuracies

    @staticmethod
    def classification(training_set, indexes, label_index):
        return NaiveBayes(training_set, indexes, label_index)


def main():

    table = remove_incomplete_rows(read_csv('auto-data.txt'))
    s = StratifiedFolds(table)
    matrix = s.stratified_k_folds(10)
    print s.get_accuracy_of_confusion(matrix)[0]
    for row in matrix:
        print row
    print_confusion(matrix)


main()
