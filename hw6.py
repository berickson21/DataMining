from hw5 import *
from hw4 import print_confusion_titanic
from copy import deepcopy
from random import shuffle, sample, choice

import numpy as numpy


class RandomForest:

    def __init__(self, training_set, att_indexes, label_index, m, n, f):

        self.training_set = deepcopy(training_set)
        for i, row in enumerate(self.training_set):  # add unique id
            row.append(i)

        self.att_indexes = att_indexes
        self.att_domains = {att: list(set(get_column(self.training_set, att))) for att in self.att_indexes}
        self.label_indexes = label_index
        self.labels = list(set(get_column(self.training_set, self.label_indexes)))
        self.num_labels = len(self.labels)

        self.remainder = []
        self.test = []
        self.partition_data()

        self.f = f  # number of attribute indexes from att_indexes to build trees on

        self.initial_forest = [self.build_tree() for _ in range(n)]  # build n instances of tree
        self.initial_forest.sort(key=lambda x: x[1], reverse=True)  # sort trees by accuracy
        self.forest = [tree[0] for tree in self.initial_forest[0:m]]  # take the m most accurate trees

        self.evaluate_ensemble()

        for i, tree in enumerate(self.forest):
            tree.save_graphviz_tree('trees/tree' + str(i))

    def build_tree(self):

        training = [choice(self.remainder) for _ in range(len(self.training_set))]
        validation = [row for row in self.remainder if row not in training]

        tree = self.get_tree(training)

        return [tree, self.evaluate_tree(tree, validation)[0]]

    def get_tree(self, training):
        pass

    def evaluate_ensemble(self):

        evaluate = self.evaluate_tree(self, self.training_set)
        self.print_matrix(evaluate[1])

    def evaluate_tree(self, tree, validation):
        pass

    def print_matrix(self, matrix):
        pass

    def get_random_training_set(self):

        table = deepcopy(self.training_set)
        shuffle(table)

        return table[:(len(table)*2)/3]

    def track_record_classify(self, instance):
        pass

    def classify(self, instance):

        classifications = [tree.classify(instance) for tree in self.forest]

        count = [[label, count_if(classifications, label)] for label in list(set(classifications))]
        count.sort(key=lambda x: x[1], reverse=True)

        return count[0][0]

    def partition_data(self):
        # make sure table is stratified
        table = deepcopy(self.training_set)
        shuffle(table)
        partition = (len(table) * 2) / 3

        self.remainder = table[0:partition]
        self.test = table[partition:]


class AutoRandomForrest(RandomForest):

    def __init__(self, training_set, att_indexes, label_index, m, n, f):
        RandomForest.__init__(self, training_set, att_indexes, label_index, m, n, f)

    def get_tree(self, training):
        return AutoDecisionTree(training, self.att_indexes, self.label_indexes, self.f, att_domains=self.att_domains)

    def evaluate_tree(self, tree, validation):
        pass

    def print_matrix(self, matrix):
        print_confusion(matrix)


class TitanicRandomForrest(RandomForest):

    def __init__(self, training_set, att_indexes, label_index, m, n, f):
        RandomForest.__init__(self, training_set, att_indexes, label_index, m, n, f)

    def get_tree(self, training):
        return TitanicDecisionTree(training, self.att_indexes, self.label_indexes, self.f, att_domains=self.att_domains)

    def evaluate_tree(self, tree, validation):

        init = [[0] * self.num_labels] * self.num_labels
        confusion = numpy.array(init)

        for instance in validation:

            c = int(tree.classify(instance))

            r = 1
            if instance[self.label_indexes] == 'yes':
                r = 0

            confusion[r][c] += 1

        matrix = numpy.matrix(confusion).tolist()

        return get_accuracy_of_confusion(matrix)[0], matrix

    def print_matrix(self, matrix):
        print_confusion_titanic(matrix)


class CancerRandomForrest(RandomForest):

    def __init__(self, training_set, att_indexes, label_index, m, n, f):
        RandomForest.__init__(self, training_set, att_indexes, label_index, m, n, f)

    def get_tree(self, training):
        pass

    def evaluate_tree(self, tree, validation):
        pass

    def print_matrix(self, matrix):
        pass


def titanic_decision_tree(table, indexes, label_index, m, n, f):  # step 1

    print_double_line('Titanic Random Forest Classifier')

    d = TitanicRandomForrest(table, indexes, label_index, m, n, f)

    for instance in sample(table, 5):
        print '\tinstance: ' + str(instance)
        print '\tclass: ' + str(convert(d.classify(instance))) + ' actual: '\
            + str(instance[3])


def auto_decision_tree(table, indexes, label_index, m, n, f):  # step 1

    print_double_line('Auto Data Random Forest Classifier')

    d = AutoRandomForrest(table, indexes, label_index, m, n, f)

    for instance in sample(table, 5):
        print '\tinstance: ' + str(instance)
        print '\tclass: ' + str(convert(d.classify(instance))) + ' actual: '\
            + str(instance[3])


def get_accuracy_of_confusion(matrix):
    total = (sum([sum(row) for row in matrix]))
    accuracies = []

    for i in range(len(matrix)):
        row = matrix[i]
        col = [r[i] for r in matrix]

        accuracies.append((total - (sum(col) + sum(row) - (2 * row[i]))) / float(total))

    return round(sum(accuracies) / float(len(accuracies)), 2), accuracies


def main():

    table = remove_incomplete_rows(read_csv('titanic.txt')[1:])
    titanic_decision_tree(table, [0, 1, 2], 3, 20, 7, 2)

    autoTable = remove_incomplete_rows(read_csv('auto-data.txt'))


main()
