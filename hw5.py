from random import sample
from math import log
from copy import deepcopy

from hw1 import get_column, get_column_as_floats
from hw3 import read_csv, remove_incomplete_rows, print_confusion, print_double_line
from hw4 import StratifiedFolds, RandomSampling, StratifiedFoldsKnn, print_confusion_titanic


class Discretization:

    def __init__(self):
        pass

    def categorize_table(self, table):

        for row in table:
            self.categorize_instance(row)

        return table

    def categorize_instance(self, row):
        pass

    @staticmethod
    def convert(value, cutoffs):

        for i, item in enumerate(cutoffs):
            if float(value) < item:
                return i + 1
            elif float(value) > cutoffs[-1]:
                return len(cutoffs) + 1


class DecisionTree(Discretization):

    def __init__(self, training_set, att_indexes, label_index):
        Discretization.__init__(self)

        self.training_set = self.categorize_table(deepcopy(training_set))
        self.training_set = training_set
        self.att_indexes = att_indexes
        self.label_index = label_index
        self.att_domains = {}
        self.att_domains = self.get_attribute_domains(
            self.training_set, self.att_indexes)
        self.decision_tree = []
        self.decision_tree = self.tdidt(
            self.training_set, self.att_indexes, self.att_domains, self.label_index)

    # [att_index_a, {value_a1: [att_index_b, {value_b1: yes, value_b2: no}], value_a2: no}]
    def tdidt(self, instances, att_indexes, att_domains, class_index):
        part_list = []
        new_att_indexes = deepcopy(att_indexes)
        new_instances = deepcopy(instances)

        # Condition 1: all instances have same class
        for att_index in att_domains:
            if self.same_class(instances, att_index):
                return part_list.append([att_index, {get_column(instances, att_index).pop()}])
        # Condition 2: no more attributes to partition
        if len(att_indexes) == 1:
            att_freqs = self.attribute_frequencies(
                instances, att_indexes[0], class_index)
            t_lst = att_freqs.items()
            lst = [list(elem) for elem in t_lst]
            maxes = [att_indexes[0], {}]
            for att_val, clss in lst:
                maxes[1].update({att_val: (max(clss[0], key=clss[0].get))})
            # print 'maxes: ' + str(maxes)
            return part_list.append(maxes)
        else:
            print 'Attribute domain: ' + str(att_domains)
            selected_att = self.select_attribute(instances, att_domains, self.label_index)
            print 'Attribute selected: ' + str(selected_att)
            new_tree = [selected_att, {self.partition_instances(instances, selected_att, att_domains)}]
            print 'NEW TREE = ' + str(new_tree)
            return part_list.append(new_tree)



    # {att_val1: part1, att_val2: part2,...}
    # att_indexes holds the attributes that have yet to be partitioned
    # att_domains holds the domains for all remaining attributes
    # # [att_index_a, {value_a1: [att_index_b, {value_b1: yes, value_b2: no}], value_a2: no}]
    def partition_instances(self, instances, att_index, att_domains):

        new_att_domains = deepcopy(att_domains)
        del new_att_domains[att_index]
        tree_list = [att_index, {v: [self.tdidt(instances, new_att_domains.keys(), new_att_domains, self.label_index)] for v in att_domains[att_index]}]
        print 'TREE LIST = ' + str(tree_list)
        # tree_list = [att_index, {v: [TitanicDecisionTree(instances, new_att_domains.keys(), self.label_index).tdidt] for v in att_domains[att_index]}]
        # print 'Tree list is: ' + str(tree_list)
        # selected_att2 = self.select_attribute(instances.remove(selected_att1), new_att_domains, self.label_index)

    def get_attribute_domains(self, instances, att_indexes):
        for index in att_indexes:
            for row in instances:
                if self.att_domains.get(index) is None:
                    self.att_domains[index] = [row[index]]
                elif self.att_domains.get(index).count(row[index]) == 0:
                    self.att_domains.get(index).append(row[index])

        return self.att_domains

    @staticmethod
    def group_by(instances, index):

        dictionary = {}
        for row in instances:
            if row[index] in dictionary:
                dictionary[row[index]].append(row)
            else:
                dictionary.update({row[index]: []})
                # print 'added key:  ' + str(row[index])
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
    def same_class(self, instances, class_index):

        label = str(instances[class_index])

        for row in instances[1:]:
            if str(row[class_index]) != label:
                return False
        return True

    # picks the attribute to partition on (smallest E_new)
    def select_attribute(self, instances, att_domains, label_index):
        e_news = dict.fromkeys(att_domains.keys())
        for att in att_domains.keys():
            e_news[att] = self.calc_enew(instances, att, label_index)
        return min(e_news, key=e_news.get)

    # takes a decision tree (produced by tdidt) and an instance to classify
    # uses the tree to predict the class label for the instance
    # returns the predicted label for the instance
    def classify(self, instance):
        pass
    #     return instances[self.label_index]

        # else:


class AutoDecisionTree (DecisionTree):

    def __init__(self, training_set, att_indexes, label_index):
        DecisionTree.__init__(self, training_set, att_indexes, label_index)
        Discretization.__init__(self)

    def categorize_instance(self, row):

        row[0] = self.convert(row[0], [13, 14, 16, 19, 23, 26, 30, 36, 44])
        row[4] = str(self.convert(row[4], [1999, 2499, 2999, 3499]))


class AutoStratifiedFolds(StratifiedFolds):

    def __init__(self, table, indexes, label_index):
        StratifiedFolds.__init__(self, table, indexes, label_index)

    def classification(self, training_set):
        return AutoDecisionTree(training_set, self.indexes, self.label_index)


class AutoRandomSampling(RandomSampling):

    def __init__(self, table, indexes, label_index, k):
        RandomSampling.__init__(self, table, indexes, label_index, k)

    def classification(self, training_set):
        return AutoDecisionTree(training_set, self.indexes, self.label_index)


class TitanicDecisionTree(DecisionTree):

    def __init__(self, training_set, att_indexes, label_index):
        DecisionTree.__init__(self, training_set, att_indexes, label_index)

    def classify(self, instance):
        return 0


class TitanicStratifiedFolds(StratifiedFoldsKnn):

    def __init__(self, table, indexes, label_index):
        StratifiedFoldsKnn.__init__(self, table, indexes, label_index)

    def classification(self, training_set):
        return TitanicDecisionTree(training_set, self.indexes, self.label_index)


def titanic_decision_tree(table, indexes, label_index):  # step 1

    print_double_line('Titanic Decision Tree Classifier')

    d = TitanicDecisionTree(table, indexes, label_index)

    # for instance in sample(table, 5):
    #     print '\tinstance: ' + str(instance)
    #     print '\tclass: ' + str(d.classify(instance)) + ' actual: '\
    #         + str(instance[3])

    # print_double_line('Titanic Decision Tree k-Folds Predictive Accuracy')

    # s = TitanicStratifiedFolds(table, indexes, label_index)

    # stratified_folds_matrix = s.stratified_k_folds(10)
    # # random_sampling = TitanicRandomSampling(table, [1, 4, 6], 0, 10)

    # stratified_folds_accuracy = s.get_accuracy_of_confusion(stratified_folds_matrix)[0]
    # # random_sampling_accuracy = random_sampling.random_sampling()

    # print '\tRandomSubsample(k=10, 2:1 Train / Test)'
    # print '\t\taccuracy = ' + str(0.71) + ', error rate = ' + str(1-0.71)
    # print '\tStratified 10-Fold Cross Validation'
    # print '\t\taccuracy = ' + str(stratified_folds_accuracy) + ', error rate = '\
    #     + str(1 - stratified_folds_accuracy)

    # print_double_line('Titanic Decision Tree Confusion Matrix Predictive Accuracy')

    # print_confusion_titanic(stratified_folds_matrix)


def auto_decision_tree(table, indexes, label_index):  # step 2

    print_double_line('Auto Decision Tree Classifier')
    d = AutoDecisionTree(table, indexes, label_index)

    # for instance in sample(table, 5):
    #     print '\tinstance: ' + str(instance)
    #     print '\tclass: ' + str(d.classify(instance)) + ' actual: '\
    #         + str(d.convert(instance[0], [13, 14, 16, 19, 23, 26, 30, 36, 44]))

    # print_double_line('Decision Tree k-Folds Predictive Accuracy')
    # d.classify(table, label_index)
    # s = AutoStratifiedFolds(table, indexes, label_index)

    print_double_line('Auto Decision Tree k-Folds Predictive Accuracy')

    # stratified_folds_matrix = s.stratified_k_folds(10)
    # random_sampling = AutoRandomSampling(table, [1, 4, 6], 0, 10)

    # stratified_folds_accuracy = s.get_accuracy_of_confusion(
    #     stratified_folds_matrix)[0]
    # random_sampling_accuracy = random_sampling.random_sampling()

    # print '\tRandomSubsample(k=10, 2:1 Train / Test)'
    # print '\t\taccuracy = ' + str(random_sampling_accuracy) + ', error rate = ' + str(0)
    # print '\tStratified 10-Fold Cross Validation'
    # print '\t\taccuracy = ' + str(stratified_folds_accuracy) + ', error rate = '\
    #     + str(1 - stratified_folds_accuracy)

    # print '\tRandomSubsample(k=10, 2:1 Train / Test)'
    # print '\t\taccuracy = ' + str(random_sampling_accuracy) + ', error rate = ' + str(1-random_sampling_accuracy)
    # print '\tStratified 10-Fold Cross Validation'
    # print '\t\taccuracy = ' + str(stratified_folds_accuracy) + ', error rate = '\
    #     + str(1 - stratified_folds_accuracy)

    # print_double_line('Auto Decision Tree Confusion Matrix Predictive Accuracy')


def main():

    table = remove_incomplete_rows(read_csv('auto-data.txt'))
    table_titanic = remove_incomplete_rows(read_csv('titanic.txt')[1:])

    # auto_decision_tree(table, [1, 4, 6], 0)
    print titanic_decision_tree(table_titanic, [0, 1, 2], 3)

main()
