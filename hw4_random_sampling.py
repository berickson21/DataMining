from copy import deepcopy
from random import shuffle
from hw2 import read_csv, remove_incomplete_rows
from hw3 import print_confusion
from hw4_Naive_Bayes import ContinuousNaiveBayes, NaiveBayes

import numpy as numpy


class RandomSampling:

    def __init__(self, table, indexes, label_index, k):
        self.table = deepcopy(table)
        self.indexes = indexes
        self.label_index = label_index
        self.labels = []

        for row in self.table:
            label = self.convert(row[self.label_index], [13, 14, 16, 19, 23, 26, 30, 36, 44])
            if label not in self.labels:
                self.labels.append(label)

        self.num_labels = len(self.labels)
        self.k = k

    def random_sampling(self):

        accuracy = []
        length = len(self.table)

        for i in range(self.k):
            table = deepcopy(self.table)
            shuffle(table)

            test_set = table[(2*length)/3:]
            training_set = table[0:(2*length)/3]

            matrix = self.construct_confusion_matrix(test_set, training_set, self.num_labels)

            accuracy.append(self.get_accuracy_of_confusion(matrix)[0])

        return sum(accuracy)/float(len(accuracy))

    def construct_confusion_matrix(self, test_set, training_set, num_labels):

        classifier = self.classification(training_set)

        init = [[0] * num_labels] * num_labels
        confusion = numpy.array(init)
        total = 0

        for instance in test_set:

            c = int(classifier.classify(instance))
            r = int(classifier.convert(instance[0], [13, 14, 16, 19, 23, 26, 30, 36, 44]))
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

    @staticmethod
    def convert(value, cutoffs):

        for i, item in enumerate(cutoffs):
            if float(value) < item:
                return i + 1
            elif float(value) > cutoffs[-1]:
                return len(cutoffs) + 1


class ContinuousRandomSampling(RandomSampling):

    def __init__(self, table, cat_indexes, cont_indexes, label_index, k):
        RandomSampling.__init__(self, table, cat_indexes, label_index, k)
        self.cont_index = cont_indexes

    def classification(self, training_set):
        return ContinuousNaiveBayes(training_set, self.indexes, self.cont_index, self.label_index)


def main():

    table = remove_incomplete_rows(read_csv('auto-data.txt'))

    r = RandomSampling(table, [1, 4, 6], 0, 10)
    cr = ContinuousRandomSampling(table, [1, 6], [4], 0, 10)
    print r.random_sampling(), cr.random_sampling()
    