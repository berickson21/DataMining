import math
from copy import deepcopy
from random import sample

import numpy

from hw1 import get_column, get_column_as_floats
from hw3 import print_double_line, read_csv, remove_incomplete_rows


class NaiveBayes:

    def __init__(self, training_set, indexes, label_index):
        self.training_set = deepcopy(training_set)
        self.indexes = indexes
        self.label_index = label_index

        self.categorize_table()
        self.labels = list(set(get_column(self.training_set, self.label_index)))
        self.initial_probabilities = [[label, len(self.group_by(self.label_index, label))\
            / float(len(training_set))] for label in self.labels]

    def classify(self, instance):

        inst = instance[:]
        self.categorize_instance(inst)

        probabilities = deepcopy(self.initial_probabilities)

        for i, label in enumerate(self.labels):
            for index in self.indexes:
                probabilities[i][1] *= self.probability(label, inst[index], index)

        probabilities.sort(key=lambda x: x[1], reverse=True)

        return probabilities[0][0]

    def probability(self, label, value, value_index):

        temp = self.group_by(self.label_index, label)

        count = 0

        for row in temp:
            if str(row[value_index]) == str(value):
                count += 1

        return count / float(len(temp))

    def categorize_table(self):

        for row in self.training_set:
            self.categorize_instance(row)

    def categorize_instance(self, row):

        row[0] = self.convert(row[0], [13, 14, 16, 19, 23, 26, 30, 36, 44])
        row[4] = str(self.convert(row[4], [1999, 2499, 2999, 3499]))

    @staticmethod
    def convert(value, cutoffs):

        for i, item in enumerate(cutoffs):
            if float(value) < item:
                return i + 1
            elif float(value) > cutoffs[-1]:
                return len(cutoffs) + 1

    def group_by(self, index, value):

        table = []

        for row in self.training_set:
            if str(row[index]) == str(value):
                table.append(row)

        return table


class ContinuousNaiveBayes(NaiveBayes):

    def __init__(self, training_set, cat_indexes, cont_indexes, label_index):
        NaiveBayes.__init__(self, training_set, cat_indexes, label_index)
        self.cont_indexes = cont_indexes

    def classify(self, instance):

        inst = instance[:]
        self.categorize_instance(inst)

        probabilities = deepcopy(self.initial_probabilities)

        for i, label in enumerate(self.labels):
            for index in self.indexes:
                probabilities[i][1] *= self.probability(label, inst[index], index)
            print probabilities
            for index in self.cont_indexes:
                probabilities[i][1] *= self.cont_probability(label, inst[index], index)

        probabilities.sort(key=lambda x: x[1], reverse=True)
        return probabilities[0][0]


    def cont_probability(self, label, value, value_index):

        table = self.group_by(0, label)
        column = get_column_as_floats(table, value_index)

        mean = sum(column)/float(len(column))
        sdev = numpy.std(column)

        first, second = 0, 0

        if sdev > 0:
            first = 1 / (math.sqrt(2 * math.pi) * sdev)
            second = math.e ** (-((float(value) - mean) ** 2) / (2 * (sdev ** 2)))
        return first * second

    def categorize_instance(self, row):

        row[0] = self.convert(row[0], [13, 14, 16, 19, 23, 26, 30, 36, 44])

def main():

    c = ContinuousNaiveBayes()

if __name__ == '__main__':
    main()
