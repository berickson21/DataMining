from copy import deepcopy
from random import choice, sample, shuffle

import numpy as numpy
from tabulate import tabulate

from hw4 import print_confusion_titanic
from hw5 import *


class RandomForest:     # Step 1

    def __init__(self, training_set, att_indexes, label_index, n, m, f):

        self.training_set = deepcopy(training_set)
        # add unique id to distinguish from other instances (Titanic)
        for i, row in enumerate(self.training_set):
            row.append(i)

        self.att_indexes = att_indexes
        self.att_domains = {
            att: list(set(get_column(self.training_set, att))) for att in self.att_indexes}
        self.label_indexes = label_index
        self.labels = sorted(
            list(set(get_column(self.training_set, self.label_indexes))))
        self.num_labels = len(self.labels)

        self.remainder = []
        self.test = []
        self.partition_data()

        self.n = n  # create n trees
        self.m = m  # keep the m most accurate trees
        self.f = f  # number of attribute indexes from att_indexes to build trees on

        self.initial_forest = [self.build_tree()
                               for _ in range(n)]     # build n instances of tree
        # sort trees by accuracy
        self.initial_forest.sort(key=lambda x: x[1], reverse=True)
        # take the m most accurate trees
        self.forest = [tree for tree in self.initial_forest[0:m]]
        # individual tree for comparison
        self.tree = self.get_normal_tree(self.remainder)

        # variables to evaluate performance
        self.confusion_matrix = None
        self.accuracy = None
        self.tree_confusion_matrix = None
        self.tree_accuracy = None

        self.evaluate_ensemble()  # evaluates ensemble and individual tree

    # builds tree
    def build_tree(self):
        training = [choice(self.remainder)
                    for _ in range(len(self.training_set))]
        validation = [row for row in self.remainder if row not in training]
        tree = self.get_tree(training)
        evaluate = self.evaluate_tree(tree, validation)

        matrix = evaluate[1]

        track = {label: [item / float(sum(get_column(matrix, i)) + .0001) for item in get_column(matrix, i)]
                 for i, label in enumerate(self.labels)}

        return [tree, evaluate[0], track]

    # returns decision tree that corresponds to data set
    def get_tree(self, training):
        pass

    def get_normal_tree(self, training):
        pass

    # evaluates ensemble and individual tree
    def evaluate_ensemble(self):

        evaluate = self.evaluate_tree(self, self.test)
        self.confusion_matrix = evaluate[1]
        self.accuracy = evaluate[0]

        evaluate = self.evaluate_tree(self.tree, self.test)
        self.tree_confusion_matrix = evaluate[1]
        self.tree_accuracy = evaluate[0]

    def evaluate_tree(self, tree, validation):
        init = [[0] * self.num_labels] * self.num_labels
        confusion = numpy.array(init)

        for instance in validation:
            location = self.get_matrix_location(tree, instance)
            c = location[0]
            r = location[1]
            confusion[r][c] += 1

        matrix = numpy.matrix(confusion).tolist()

        return get_accuracy_of_confusion(matrix)[0], matrix

    def get_matrix_location(self, tree, instance):
        pass

    def print_matrix(self):
        pass

    # randomly partitions data into two sets
    def get_random_training_set(self):

        table = deepcopy(self.training_set)
        shuffle(table)

        return table[:(len(table) * 2) / 3]

    def track_record_classify(self, instance):

        classifications = [
            tree[2][self.check(tree[0].classify(instance))] for tree in self.forest]

        count = [[label, 0] for label in self.labels]

        for row in classifications:
            for i, item in enumerate(row):
                count[i][1] += item

        count.sort(key=lambda x: x[1], reverse=True)

        return count[0][0]

    @staticmethod
    def check(value):
        return value

    def classify(self, instance):

        classifications = [tree[0].classify(instance) for tree in self.forest]

        count = [[label, count_if(classifications, label)]
                 for label in list(set(classifications))]

        # sort labels based on votes from greatest to lowest
        count.sort(key=lambda x: x[1], reverse=True)

        return count[0][0]  # return value with the most votes

    def partition_data(self):
        # make sure table is stratified
        table = deepcopy(self.training_set)
        shuffle(table)
        partition = (len(table) * 2) / 3

        self.remainder = table[0:partition]
        self.test = table[partition:]


class AutoRandomForest(RandomForest, Discretization):

    def __init__(self, training_set, att_indexes, label_index, n, m, f):
        RandomForest.__init__(self, self.categorize_table(
            training_set), att_indexes, label_index, n, m, f)

    def get_tree(self, training):
        return AutoDecisionTree(training, self.att_indexes, self.label_indexes, self.f, att_domains=self.att_domains)

    def get_normal_tree(self, training):
        return AutoDecisionTree(training, self.att_indexes, self.label_indexes, len(self.att_indexes))

    def get_matrix_location(self, tree, instance):
        c = int(tree.classify(instance)) - 1
        r = int(instance[self.label_indexes]) - 1

        return c, r

    def print_matrix(self):
        print_confusion(self.confusion_matrix)

    def print_individual_matrix(self):
        print_confusion(self.tree_confusion_matrix)


class TitanicRandomForest(RandomForest):

    def __init__(self, training_set, att_indexes, label_index, n, m, f):
        RandomForest.__init__(self, training_set,
                              att_indexes, label_index, n, m, f)

    def get_tree(self, training):
        return TitanicDecisionTree(training, self.att_indexes, self.label_indexes, self.f, att_domains=self.att_domains)

    def get_normal_tree(self, training):
        return TitanicDecisionTree(training, self.att_indexes, self.label_indexes, len(self.att_indexes))

    @staticmethod
    def check(value):
        if value == 0:
            return 'yes'
        else:
            return 'no'

    def get_matrix_location(self, tree, instance):
        c = int(tree.classify(instance))

        r = 1
        if instance[self.label_indexes] == 'yes':
            r = 0

        return c, r

    def print_matrix(self):
        print_confusion_titanic(self.confusion_matrix)

    def print_individual_matrix(self):
        print_confusion_titanic(self.tree_confusion_matrix)


class WisconsinRandomForest(RandomForest):

    def __init__(self, training_set, att_indexes, label_index, n, m, f):
        RandomForest.__init__(self, training_set,
                              att_indexes, label_index, n, m, f)

    def get_tree(self, training):
        return WisconsinDecisionTree(training, self.att_indexes, self.label_indexes, self.f, att_domains=self.att_domains)

    def get_normal_tree(self, training):
        return WisconsinDecisionTree(training, self.att_indexes, self.label_indexes, len(self.att_indexes))

    def get_matrix_location(self, tree, instance):

        c = 0
        if int(tree.classify(instance)) == 2:
            c = 1
        r = 0
        if int(instance[self.label_indexes]) == 2:
            r = 1

        return c, r

    @staticmethod
    def print_matrix_wisconsin(matrix):

        accuracies = get_accuracy_of_confusion(matrix)[1]

        for i, row in enumerate(matrix):
            if i == 0:
                row.insert(0, 'benign')
            else:
                row.insert(0, 'malignant')
            row.append(sum(row[1:]))
            row.append(round(accuracies[i], 2))

        headers = ['Tumor', 'benign', 'malignant', 'Total', 'Recognition(%)']

        print tabulate(matrix, headers=headers, tablefmt="rst")

    def print_matrix(self):
        self.print_matrix_wisconsin(self.confusion_matrix)

    def print_individual_matrix(self):
        self.print_matrix_wisconsin(self.tree_confusion_matrix)


class WisconsinRandomForestTrackRecord(WisconsinRandomForest):

    def __init__(self, training_set, att_indexes, label_index, n, m, f):
        RandomForest.__init__(self, training_set,
                              att_indexes, label_index, n, m, f)

    def get_matrix_location_track(self, instance):

        c = 0
        if int(self.track_record_classify(instance)) == 2:
            c = 1
        r = 0
        if int(instance[self.label_indexes]) == 2:
            r = 1

        return c, r

# evaluates ensemble and individual tree
    def evaluate_ensemble(self):

        evaluate = self.evaluate_tree_track(self, self.test)
        self.confusion_matrix = evaluate[1]
        self.accuracy = evaluate[0]

        evaluate = self.evaluate_tree(self.tree, self.test)
        self.tree_confusion_matrix = evaluate[1]
        self.tree_accuracy = evaluate[0]

    def evaluate_tree_track(self, tree, validation):
        init = [[0] * self.num_labels] * self.num_labels
        confusion = numpy.array(init)

        for instance in validation:
            location = self.get_matrix_location_track(instance)
            c = location[0]
            r = location[1]
            confusion[r][c] += 1

        matrix = numpy.matrix(confusion).tolist()

        return get_accuracy_of_confusion(matrix)[0], matrix


class TitanicRandomForestTrackRecord(TitanicRandomForest):

    def __init__(self, training_set, att_indexes, label_index, n, m, f):
        TitanicRandomForest.__init__(
            self, training_set, att_indexes, label_index, n, m, f)

    def get_matrix_location_track(self, instance):
        c = int(self.reverse(self.track_record_classify(instance)))

        r = 1
        if instance[self.label_indexes] == 'yes':
            r = 0

        return c, r

    @staticmethod
    def check(value):

        if value == 0:
            return 'yes'
        else:
            return 'no'

    @staticmethod
    def reverse(value):

        if value == 'yes':
            return 0
        else:
            return 1

    # evaluates ensemble and individual tree
    def evaluate_ensemble(self):

        evaluate = self.evaluate_tree_track(self, self.test)
        self.confusion_matrix = evaluate[1]
        self.accuracy = evaluate[0]

        evaluate = self.evaluate_tree(self.tree, self.test)
        self.tree_confusion_matrix = evaluate[1]
        self.tree_accuracy = evaluate[0]

    def evaluate_tree_track(self, tree, validation):
        init = [[0] * self.num_labels] * self.num_labels
        confusion = numpy.array(init)

        for instance in validation:
            location = self.get_matrix_location_track(instance)
            c = location[0]
            r = location[1]
            confusion[r][c] += 1

        matrix = numpy.matrix(confusion).tolist()

        return get_accuracy_of_confusion(matrix)[0], matrix


class AutoRandomForestTrackRecord(AutoRandomForest):

    def __init__(self, training_set, att_indexes, label_index, n, m, f):
        AutoRandomForest.__init__(
            self, training_set, att_indexes, label_index, n, m, f)

    def get_matrix_location_track(self, instance):
        c = int(self.track_record_classify(instance)) - 1
        r = int(instance[self.label_indexes]) - 1

        return c, r

    # evaluates ensemble and individual tree
    def evaluate_ensemble(self):

        evaluate = self.evaluate_tree_track(self, self.test)
        self.confusion_matrix = evaluate[1]
        self.accuracy = evaluate[0]

        evaluate = self.evaluate_tree(self.tree, self.test)
        self.tree_confusion_matrix = evaluate[1]
        self.tree_accuracy = evaluate[0]

    def evaluate_tree_track(self, tree, validation):
        init = [[0] * self.num_labels] * self.num_labels
        confusion = numpy.array(init)

        for instance in validation:
            location = self.get_matrix_location_track(instance)
            c = location[0]
            r = location[1]
            confusion[r][c] += 1

        matrix = numpy.matrix(confusion).tolist()

        return get_accuracy_of_confusion(matrix)[0], matrix


class WisconsinDecisionTree(DecisionTree):

    def __init__(self, training_set, att_indexes, label_index, f, **kwargs):
        DecisionTree.__init__(self, training_set,
                              att_indexes, label_index, f, **kwargs)

        column_names = ['clump thickness', 'cell size', 'cell shape', 'marginal adhesion', 'epithelial size',
                        'bare nuclei', 'bland chromatin', 'normal nucleoli', 'mitoses', 'tumor']

        DisplayTree.__init__(self, self.decision_tree, column_names)

    def categorize_instance(self, row):
        return None


def random_forest(classifier, class_name):

    print_double_line(class_name + ' Random Forest Classifier')

    print_double_line(class_name + ' Individual Tree Confusion Matrix')
    classifier.print_individual_matrix()

    print_double_line(class_name + ' Random Forest Confusion Matrix:  n = ' + str(classifier.n) +
                      ' m = ' + str(classifier.m) + ' f = ' + str(classifier.f))
    classifier.print_matrix()

    print_double_line(class_name + ' Predictive accuracy: n = ' + str(classifier.n) + ' m = ' + str(classifier.m)
                      + ' f = ' + str(classifier.f))

    print '\tRandom Forest'
    print '\t\tAccuracy: ' + str(classifier.accuracy)
    print '\tIndividual Tree'
    print '\t\tAccuracy: ' + str(classifier.tree_accuracy)


def get_accuracy_of_confusion(matrix):
    total = (sum([sum(row) for row in matrix]))
    accuracies = []

    for i in range(len(matrix)):
        row = matrix[i]
        col = [r[i] for r in matrix]

        accuracies.append(
            (total - (sum(col) + sum(row) - (2 * row[i]))) / float(total))

    return round(sum(accuracies) / float(len(accuracies)), 2), accuracies


def random_forest_param_testing(table, auto_table, wisconsin_table):
    print_double_line(
        'Testing for best n, m, f params. This could take a while...')
    n_list = [20, 30, 40]
    m_list = [5, 10, 15]
    f_list = [2, 3]

    # each best vals list contains: [accuracy, n, m, f]
    best_vals_titanic = [0, 0, 0, 0]
    best_vals_auto = [0, 0, 0, 0]
    best_vals_wisconsin = [0, 0, 0, 0]

    for n in n_list:
        for m in m_list:
            for f in f_list:
                titanic_accuracy_list = []
                auto_accuracy_list = []
                wisconsin_accuracy_list = []
                for i in range(4):
                    titanic_accuracy_list.append(TitanicRandomForest(
                        table, [0, 1, 2], 3, n, m, f).accuracy)
                    auto_accuracy_list.append(AutoRandomForest(
                        auto_table, [1, 4, 6], 0, n, m, f).accuracy)
                    wisconsin_accuracy_list.append(WisconsinRandomForest(
                        wisconsin_table, [0, 1, 2, 3, 4, 5, 6, 7, 8], 9, n, m, f).accuracy)

                # Gets the average accuracy of five runs with each set of n, m,
                # and f vals.
                titanic_accuracy = sum(
                    titanic_accuracy_list) / float(len(titanic_accuracy_list))
                auto_accuracy = sum(auto_accuracy_list) / \
                    float(len(auto_accuracy_list))
                wisconsin_accuracy = sum(
                    wisconsin_accuracy_list) / float(len(wisconsin_accuracy_list))

                if titanic_accuracy > best_vals_titanic[0]:
                    best_vals_titanic = [titanic_accuracy, n, m, f]

                if auto_accuracy > best_vals_auto[0]:
                    best_vals_auto = [auto_accuracy, n, m, f]

                if wisconsin_accuracy > best_vals_wisconsin[0]:
                    best_vals_wisconsin = [wisconsin_accuracy, n, m, f]

    print 'Best Titanic Accuracy: ' + str(best_vals_titanic[0])
    print 'Parameters n, m, f: ' + str(best_vals_titanic[1:])
    print ''
    print 'Best Auto Data Accuracy: ' + str(best_vals_auto[0])
    print 'Parameters n, m, f: ' + str(best_vals_auto[1:])
    print ''
    print 'Best Wisconsin Accuracy: ' + str(best_vals_wisconsin[0])
    print 'Parameters n, m, f: ' + str(best_vals_wisconsin[1:])


def main():

    table = remove_incomplete_rows(read_csv('titanic.txt')[1:])
    auto_table = remove_incomplete_rows(read_csv('auto-data.txt'))
    wisconsin_table = remove_incomplete_rows(read_csv('wisconsin.txt'))

    # Step 2
    random_forest(TitanicRandomForest(
        table, [0, 1, 2], 3, 20, 7, 2), 'Titanic')
    random_forest(AutoRandomForest(
        auto_table, [1, 4, 6], 0, 20, 7, 2), 'Auto Data')

    # Step 3
    random_forest_param_testing(table, auto_table, wisconsin_table)

    # Step 4
    random_forest(WisconsinRandomForest(wisconsin_table, [
        0, 1, 2, 3, 4, 5, 6, 7, 8], 9, 20, 7, 3), 'Wisconsin')

    # # Extra Credit
    random_forest(TitanicRandomForestTrackRecord(
        table, [0, 1, 2], 3, 20, 7, 2), 'Track Record Titanic')

    random_forest(AutoRandomForestTrackRecord(
        auto_table, [1, 4, 6], 0, 20, 7, 2), 'Track Record Auto Data')

    random_forest(WisconsinRandomForestTrackRecord(wisconsin_table, [
        i for i in range(8)], 9, 20, 7, 3), 'Track Record Wisconsin')

main()
