from hw5 import *
from hw4 import print_confusion_titanic
from copy import deepcopy
from random import shuffle, sample, choice
from tabulate import tabulate
import numpy as numpy


class RandomForest:     # Step 1

    def __init__(self, training_set, att_indexes, label_index, m, n, f):

        self.training_set = deepcopy(training_set)
        for i, row in enumerate(self.training_set):  # add unique id to distinguish from other instances (Titanic)
            row.append(i)

        self.att_indexes = att_indexes
        self.att_domains = {att: list(set(get_column(self.training_set, att))) for att in self.att_indexes}
        self.label_indexes = label_index
        self.labels = sorted(list(set(get_column(self.training_set, self.label_indexes))))
        self.num_labels = len(self.labels)

        self.remainder = []
        self.test = []
        self.partition_data()

        self.n = n  # create n trees
        self.m = m  # keep the m most accurate trees
        self.f = f  # number of attribute indexes from att_indexes to build trees on

        self.initial_forest = [self.build_tree() for _ in range(n)]     # build n instances of tree
        self.initial_forest.sort(key=lambda x: x[1], reverse=True)      # sort trees by accuracy
        self.forest = [tree for tree in self.initial_forest[0:m]]    # take the m most accurate trees
        self.tree = self.get_normal_tree(self.remainder)                # individual tree for comparison

        # variables to evaluate performance
        self.confusion_matrix = None
        self.accuracy = None
        self.tree_confusion_matrix = None
        self.tree_accuracy = None

        self.evaluate_ensemble()  # evaluates ensemble and individual tree

    # builds tree
    def build_tree(self):
        training = [choice(self.remainder) for _ in range(len(self.training_set))]
        validation = [row for row in self.remainder if row not in training]
        tree = self.get_tree(training)
        evaluate = self.evaluate_tree(tree, validation)

        matrix = evaluate[1]

        track = {label: [item/float(sum(get_column(matrix, i))+.0001) for item in get_column(matrix, i)]
                 for i, label in enumerate(self.labels)}

        return [tree, evaluate[0], track]

    # returns decision tree that corresponds to data set
    def get_tree(self, training):
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

        return table[:(len(table)*2)/3]

    def track_record_classify(self, instance):

        classifications = [tree[2][self.check(tree[0].classify(instance))] for tree in self.forest]

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

        count = [[label, count_if(classifications, label)] for label in list(set(classifications))]

        count.sort(key=lambda x: x[1], reverse=True)  # sort labels based on votes from greatest to lowest

        return count[0][0]  # return value with the most votes

    def partition_data(self):
        # make sure table is stratified
        table = deepcopy(self.training_set)
        shuffle(table)
        partition = (len(table) * 2) / 3

        self.remainder = table[0:partition]
        self.test = table[partition:]


class AutoRandomForrest(RandomForest, Discretization):

    def __init__(self, training_set, att_indexes, label_index, m, n, f):
        RandomForest.__init__(self, self.categorize_table(training_set), att_indexes, label_index, m, n, f)

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


class TitanicRandomForrest(RandomForest):

    def __init__(self, training_set, att_indexes, label_index, m, n, f):
        RandomForest.__init__(self, training_set, att_indexes, label_index, m, n, f)

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


class WisconsinRandomForrest(RandomForest):

    def __init__(self, training_set, att_indexes, label_index, m, n, f):
        RandomForest.__init__(self, training_set, att_indexes, label_index, m, n, f)

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


class WisconsinRandomForrestTrackRecord(WisconsinRandomForrest):

    def __init__(self, training_set, att_indexes, label_index, m, n, f):
        RandomForest.__init__(self, training_set, att_indexes, label_index, m, n, f)

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


class TitanicRandomForrestTrackRecord(TitanicRandomForrest):

    def __init__(self, training_set, att_indexes, label_index, m, n, f):
        TitanicRandomForrest.__init__(self, training_set, att_indexes, label_index, m, n, f)

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


class AutoRandomForrestTrackRecord(AutoRandomForrest):

    def __init__(self, training_set, att_indexes, label_index, m, n, f):
        AutoRandomForrest.__init__(self,training_set, att_indexes, label_index, m, n, f)

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
        DecisionTree.__init__(self, training_set, att_indexes, label_index, f, **kwargs)

        column_names = ['clump thickness', 'cell size', 'cell shape', 'marginal adhesion', 'epithelial size',
                        'bare nuclei', 'bland chromatin', 'normal nucleoli', 'mitoses', 'tumor']

        DisplayTree.__init__(self, self.decision_tree, column_names)

    def categorize_instance(self, row):
        return None


def random_forrest(classifier, class_name):

    print_double_line(class_name + ' Random Forest Classifier')

    print_double_line(class_name + ' Individual Tree Confusion Matrix')
    classifier.print_individual_matrix()

    print_double_line(class_name + ' Random Forrest Confusion Matrix:  n = ' + str(classifier.n) +
                      ' m = ' + str(classifier.m) + ' f = ' + str(classifier.f))
    classifier.print_matrix()

    print_double_line(class_name + ' Predictive accuracy: n = ' + str(classifier.n) + ' m = ' + str(classifier.m)
                      + ' f = ' + str(classifier.f))

    print '\tRandom Forrest'
    print '\t\tAccuracy: ' + str(classifier.accuracy)
    print '\tIndividual Tree'
    print '\t\tAccuracy: ' + str(classifier.tree_accuracy)


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
    auto_table = remove_incomplete_rows(read_csv('auto-data.txt'))
    wisconsin_table = remove_incomplete_rows(read_csv('wisconsin.txt'))

    # Step 2
    random_forrest(TitanicRandomForrest(table, [0, 1, 2], 3, 20, 7, 2), 'Titanic')
    random_forrest(AutoRandomForrest(auto_table, [1, 4, 6], 0, 20, 7, 2), 'Auto Data')

    # Step 4
    random_forrest(WisconsinRandomForrest(wisconsin_table, [0, 1, 2, 3, 4, 5, 6, 7, 8], 9, 20, 7, 3), 'Wisconsin')

    # Extra Credit
    random_forrest(TitanicRandomForrestTrackRecord(table, [0, 1, 2], 3, 20, 7, 2), 'Track Record Titanic')
    random_forrest(AutoRandomForrestTrackRecord(auto_table, [1, 4, 6], 0, 20, 7, 2), 'Track Record Auto Data')
    random_forrest(WisconsinRandomForrestTrackRecord(wisconsin_table, [i for i in range(8)]
                                                     , 9, 20, 7, 3), 'Track Record Wisconsin')

main()
