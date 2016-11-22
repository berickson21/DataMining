from random import sample
from math import log
from copy import deepcopy
from graphviz import Graph

from hw1 import get_column, get_column_as_floats
from hw3 import read_csv, remove_incomplete_rows, print_confusion, print_double_line, COLUMN_NAMES
from hw4 import StratifiedFolds, RandomSampling, StratifiedFoldsKnn, print_confusion_titanic, random_sampling


class Discretization:

    def __init__(self):
        pass

    def categorize_table(self, table):

        for row in table:
            self.categorize_instance(row)

        return table

    def categorize_instance(self, row):

        row[0] = str(self.convert(row[0], [13, 14, 16, 19, 23, 26, 30, 36, 44]))
        # print self.convert(row[4], [1999, 2499, 2999, 3499])
        row[4] = str(self.convert(row[4], [1999, 2499, 2999, 3499]))

    @staticmethod
    def convert(value, cutoffs):

        for i, item in enumerate(cutoffs):
            if float(value) < item:
                return i + 1
            elif float(value) > cutoffs[-1]:
                return len(cutoffs) + 1


class DisplayTree:

    def __init__(self, decision_tree, column_names):
        self.decision_tree = decision_tree
        self.attribute_names = column_names

        self.dot = None

    # recursive helper function
    def save_graphviz_tree(self, file_name):
        self.dot = Graph()
        self.dot.graph_attr['rankdir'] = 'LR'
        self.create_dot_graph(self.decision_tree)
        self.dot.render(file_name)

    def create_dot_graph(self, tree, head_name='', number=0):

        if number == 0 and isinstance(tree, list):
            name = 'h' + str(number)
            self.dot.node(name, 'Attribute: ' + str(self.attribute_names[int(tree[0])]), shape='box')
            number += 1
            self.create_dot_graph(tree[1], name, number)

        elif isinstance(tree, list):
            name = head_name + str(self.attribute_names[int(tree[0])]) + str(number)
            self.dot.node(name, 'Attribute: ' + str(self.attribute_names[int(tree[0])]), shape='box')
            self.dot.edge(head_name, name)
            number += 1
            self.create_dot_graph(tree[1], name, number)

        elif isinstance(tree, dict):

            for key in tree:
                name = head_name + str(key) + str(number)
                self.dot.node(name, 'Value: ' + str(key))
                self.dot.edge(head_name, name)
                number += 1
                self.create_dot_graph(tree[key], name, number)
        else:
            name = head_name + str(tree) + str(number)
            self.dot.node(name, tree)
            self.dot.edge(head_name, name)
            number += 1

    def print_if_statements(self):
        self.print_ifs(self.decision_tree)

    # recursive helper function
    def print_ifs(self, tree, statement='if '):

        if isinstance(tree, list):
            state = statement + str(self.attribute_names[int(tree[0])]) + ' == '
            self.print_ifs(tree[1], state)

        elif isinstance(tree, dict):
            for key in tree:
                self.print_ifs(tree[key], statement + str(key) + ' and ')

        else:
            print statement[:-5] + ' then label is ' + tree


class DecisionTree(DisplayTree, Discretization):

    def __init__(self, training_set, att_indexes, label_index, f, **kwargs):
        Discretization.__init__(self)
        self.training_set = deepcopy(training_set)
        self.att_indexes = att_indexes
        self.label_index = label_index
        self.f = f

        if 'att_domains' in kwargs:
            self.att_domains = kwargs['att_domains']
        else:
            self.att_domains = {att: list(set(get_column(self.training_set, att))) for att in att_indexes}

        self.decision_tree = self.tdidt(self.training_set, self.att_indexes)

    # [att_index_a, {value_a1: [att_index_b, {value_b1: yes, value_b2: no}], value_a2: no}]
    def tdidt(self, instances, att_indexes):

        if self.same_labels(instances):

            return instances[0][self.label_index]

        elif len(instances) == 0:

            return 'go back you screwed up'

        elif len(att_indexes) == 0:

            probabilities = self.get_probabilities(instances)
            return probabilities[0][0]

        else:

            indexes = att_indexes[:]

            if len(indexes) <= self.f:
                part_index = self.select_attribute(instances, indexes)
            else:
                part_index = self.select_attribute(instances, sample(indexes, self.f))
            partitions = self.group_by(instances, part_index)
            indexes.remove(part_index)

            dictionary = {}

            for key in partitions:

                td = self.tdidt(partitions[key], indexes)

                if td == 'go back you screwed up':
                    probabilities = self.get_probabilities(instances)
                    dictionary.update({key: probabilities[0][0]})
                else:
                    dictionary.update({key: td})

            return [str(part_index), dictionary]

    def get_probabilities(self, instances):
        column = get_column(instances, self.label_index)
        labels = list(set(column))

        probabilities = [[label, self.count_if(column, label)] for label in labels]
        probabilities.sort(key=lambda x: x[1], reverse=True)
        return probabilities

    @staticmethod
    def count_if(column, label):

        count = 0
        for item in column:
            if str(item) == str(label):
                count += 1
        return count

    def group_by(self, instances, index):

        dictionary = {att: [] for att in self.att_domains[index]}

        for row in instances:
            dictionary[row[index]].append(row)

        return dictionary

    # calculates enew using entropy stuff.
    def calc_enew(self, instances, att_index, class_index):

        # get the length of the partition
        D = len(instances)

        # calculate the partition stats for att_index (see below)
        freqs = self.attribute_frequencies(instances, att_index, class_index)
        # print freqs.keys()
        # find E_new from freqs (calc weighted avg)
        E_new = 0
        for att_val in freqs:
            D_j = float(freqs[att_val][1])
            probs = [(c / D_j) for (_, c) in freqs[att_val][0].items()]

            E_D_j = 0
            for p in probs:
                if p != 0:
                    E_D_j += p * log(p, 2)
            E_D_j *= -1
            E_new += (D_j / D) * E_D_j
        return E_new

    # Returns the class frequencies for each attribute value:
    # {att_val:[{class1: freq, class2: freq, ...}, total], ...}
    def attribute_frequencies(self, instances, att_index, class_index):

        # get unique list of attribute and class values
        # print "att index: " + str([row[1] for row in instances])
        att_vals = list(set(get_column(instances, att_index)))
        class_vals = list(set(get_column(instances, class_index)))

        # initialize the result
        result = {v: [{c: 0 for c in class_vals}, 0] for v in att_vals}
        # build up the frequencies
        # iterator = 0
        for row in instances:
            label = row[class_index]
            att_val = row[att_index]
            result[att_val][0][label] += 1
            result[att_val][1] += 1
        return result

    # Returns true if all instances have same label
    def same_labels(self, instances):
        return len(list(set(get_column(instances, self.label_index)))) == 1

    # picks the attribute to partition on (smallest E_new)
    def select_attribute(self, instances, att_domains):

        e_news = {key: self.calc_enew(instances, key, self.label_index) for key in att_domains}

        return min(e_news, key=e_news.get)

    # takes a decision tree (produced by tdidt) and an instance to classify
    # uses the tree to predict the class label for the instance
    # returns the predicted label for the instance
    def classify(self, instance):
        inst = instance[:]
        self.categorize_instance(inst)

        label = self.classify_recursive(self.decision_tree, inst)

        if str(label) == 'yes':
            return 0
        elif str(label) == 'no':
            return 1
        else:
            return label

    # recursive helper function
    def classify_recursive(self, tree, instance):

        if isinstance(tree, list):
            if str(instance[int(tree[0])]) not in tree[1]:
                return 1
            else:
                return self.classify_recursive(tree[1][str(instance[int(tree[0])])], instance)
        else:
            return tree


class AutoDecisionTree (Discretization, DecisionTree):

    def __init__(self, training_set, att_indexes, label_index, f, **kwargs):
        DecisionTree.__init__(self, training_set, att_indexes, label_index, f, **kwargs)
        DisplayTree.__init__(self, self.decision_tree, COLUMN_NAMES)


class AutoStratifiedFolds(StratifiedFolds):

    def __init__(self, table, indexes, label_index):
        StratifiedFolds.__init__(self, table, indexes, label_index)

    def classification(self, training_set):
        return AutoDecisionTree(training_set, self.indexes, self.label_index, len(self.indexes))

    def categorize_instance(self, row):
        pass


class AutoRandomSampling(RandomSampling):

    def __init__(self, table, indexes, label_index, k):
        RandomSampling.__init__(self, table, indexes, label_index, k)

    def classification(self, training_set):
        return AutoDecisionTree(training_set, self.indexes, self.label_index, len(self.indexes))


class TitanicDecisionTree(DecisionTree):

    def __init__(self, training_set, att_indexes, label_index, f, **kwargs):

        DecisionTree.__init__(self, training_set, att_indexes, label_index, f, **kwargs)
        DisplayTree.__init__(self, self.decision_tree, ['Class', 'Age', 'Sex', 'Survived'])

    def categorize_instance(self, row):
        return None


class TitanicStratifiedFolds(StratifiedFoldsKnn):

    def __init__(self, table, indexes, label_index):
        StratifiedFoldsKnn.__init__(self, table, indexes, label_index)

    def classification(self, training_set):
        return TitanicDecisionTree(training_set, self.indexes, self.label_index, len(self.indexes))


def titanic_decision_tree(table, indexes, label_index):  # step 1

    print_double_line('Titanic Decision Tree Classifier')

    d = TitanicDecisionTree(table, indexes, label_index, len(indexes))

    for instance in sample(table, 5):
        print '\tinstance: ' + str(instance)
        print '\tclass: ' + str(convert(d.classify(instance))) + ' actual: '\
            + str(instance[3])

    print_double_line('Titanic Decision Tree k-Folds Predictive Accuracy')

    s = TitanicStratifiedFolds(table, indexes, label_index)

    stratified_folds_matrix = s.stratified_k_folds(10)

    stratified_folds_accuracy = s.get_accuracy_of_confusion(stratified_folds_matrix)[0]
    random_sampling_accuracy = random_sampling

    print '\tRandomSubsample(k=10, 2:1 Train / Test)'
    print '\t\taccuracy = ' + str(random_sampling_accuracy) + ', error rate = ' + str(1-random_sampling_accuracy)
    print '\tStratified 10-Fold Cross Validation'
    print '\t\taccuracy = ' + str(stratified_folds_accuracy) + ', error rate = '\
        + str(1 - stratified_folds_accuracy)

    print_double_line('Titanic Decision Tree Confusion Matrix Predictive Accuracy')

    print_confusion_titanic(stratified_folds_matrix)

    print_double_line('Titanic If Statements')

    d.print_if_statements()

    d.save_graphviz_tree('trees/titanic_decision_tree')


def convert(value):
    if value == 1:
        return 'no'
    else:
        return 'yes'


def auto_decision_tree(table, indexes, label_index):  # step 2

    print_double_line('Auto Decision Tree Classifier')
    d = AutoDecisionTree(table, indexes, label_index, len(indexes))

    for instance in sample(table, 5):
        print '\tinstance: ' + str(instance)
        print '\tclass: ' + str(d.classify(instance)) + ' actual: '\
            + str(d.convert(instance[0], [13, 14, 16, 19, 23, 26, 30, 36, 44]))

    print_double_line('Auto Decision Tree k-Folds Predictive Accuracy')

    s = AutoStratifiedFolds(table, indexes, label_index)

    stratified_folds_matrix = s.stratified_k_folds(10)
    random_sampling = AutoRandomSampling(table, [1, 4, 6], 0, 10)

    stratified_folds_accuracy = s.get_accuracy_of_confusion(stratified_folds_matrix)[0]
    random_sampling_accuracy = random_sampling.random_sampling()

    print '\tRandomSubsample(k=10, 2:1 Train / Test)'
    print '\t\taccuracy = ' + str(random_sampling_accuracy) + ', error rate = ' + str(1 - random_sampling_accuracy)
    print '\tStratified 10-Fold Cross Validation'
    print '\t\taccuracy = ' + str(stratified_folds_accuracy) + ', error rate = ' + str(1 - stratified_folds_accuracy)

    print_double_line('Auto Decision Tree Confusion Matrix Predictive Accuracy')

    print_confusion(stratified_folds_matrix)

    print_double_line('Auto Decision If Statements')

    d.print_if_statements()

    d.save_graphviz_tree('trees/auto_data_decision_tree')


def count_if(column, label):
    count = 0
    for item in column:
        if str(item) == str(label):
            count += 1
    return count

