from random import sample
from math import log
from copy import deepcopy

from hw1 import get_column
from hw3 import read_csv, remove_incomplete_rows, print_confusion, print_double_line
from hw4 import StratifiedFolds, RandomSampling


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
        self.decision_tree = {}
        self.att_domains = {}

        self.att_domains = self.get_attribute_domains(self.training_set, self.att_indexes)
        # print self.select_attribute(self.training_set, self.att_indexes, self.att_domains, self.label_index)

    # instances is the current partition
    # att_indexes are the indexes of attributes used for classification
    # att_domains is the possible values for each attribute (by index)
    # label_index is the attribute used as the class label
    # def tdit(self, instances, att_indexes, att_domains, label_index):    
        

        

    # def create_decision_tree(self, sofar, todo, label_index):

        
    #     self.decision_tree = self.group_by(self.training_set, label_index)

    #     for key in self.decision_tree:
    #         self.decision_tree[key] = self.group_by(self.decision_tree[key], 1)

    #   for key in self.decision_tree:
    #       for row in self.decision_tree[key]:
    #           print row

    # Creates a dictionary with all of the occurances
    # of each value for a given attribute (column).
    
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
    def calc_enew(self, instances, att_index, label_index):

        # get the length of the partition
        D = len(instances)
       
        # calculate the partition stats for att_index (see below)
        freqs = self.attribute_frequencies(instances, att_index, label_index)
       
        # find E_new from freqs (calc weighted avg)
        E_new = 0
        for att_val in freqs:
            D_j = float(freqs[att_val][1])
            probs = [(c/D_j) for (_, c) in freqs[att_val][0].items()]
            # print probs
            E_D_j = -sum([p*log(p,2) for p in probs])
            E_new += (D_j/D)*E_D_j
        return E_new

    # Returns the class frequencies for each attribute value:
    # {att_val:[{class1: freq, class2: freq, ...}, total], ...}
    def attribute_frequencies(self, instances, att_index, label_index):
        pass
        # # get unique list of attribute and class values
        # att_vals = list(set(get_column(instances, att_index)))
        # class_vals = list(set(get_column(instances, label_index)))

        # # initialize the result
        # result = {v: [{c: 0 for c in class_vals}, 0] for v in att_vals}
        
        
        # # build up the frequencies
        # for row in instances:
        #     label = row[label_index]
        #     att_val = row[att_index]
        #     result[att_val][0][label] += 1
        #     result[att_val][1] += 1
        # return result


    # Returns true if all instances have same label
    # def same_class(instances, label_index):

    #     label = str(instances[self.label_index])

    #     for row in instances[1:]:
    #         if str(row[self.label_index]) != label:
    #             return False
    #     return True

    # List of stats: [[label1, occ1, total1], [label2, occ2, total2]...
    def partition_stats(self, instances, label_index):
        pass

    # {att_val1: part1, att_val2: part2,...}
    def partition_instances(self, instances, att_indexes, att_domains):
        part_list = {}
        for value in att_domains:
            part_list.update({value: DecisionTree(instances, att_indexes, self.label_index)})
        return part_list

    # picks the attribute to partition on (smallest E_new)
    def select_attribute(self, instances, att_indexes, att_domains, label_index):
        e_new = dict.fromkeys(self.att_domains.keys())
        for index in att_indexes:
            print index
            print self.calc_enew(instances, index, self.label_index)
        # print e_new
        # return min(e_new, k=e_new.get)
        # result = []
        # min_value = None
        # for key, value in e_new.iteritems():
        #     if min_value is None or value < min_value:
        #         min_value = value
        #         result = []
        #     if value == min_value:
        #         result.append(key)

        # return result
        # for attribute in att_indexes:


   
    # takes a decision tree (produced by tdidt) and an instance to classify
    # uses the tree to predict the class label for the instance
    # returns the predicted label for the instance
    # def classify(self, instances, instance):

    #     return instances[self.label_index]
        # if self.same_class(instances, self.label_index):
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


def auto_decision_tree(table, indexes, label_index):  # step 1

    print_double_line('Decision Tree Classifier')
    d = AutoDecisionTree(table, indexes, label_index)


    # for instance in sample(table, 5):
    #     print '\tinstance: ' + str(instance)
    #     print '\tclass: ' + str(d.classify(instance)) + ' actual: '\
    #         + str(d.convert(instance[0], [13, 14, 16, 19, 23, 26, 30, 36, 44]))

    # print_double_line('Decision Tree k-Folds Predictive Accuracy')
    # d.classify(table, label_index)
    # s = AutoStratifiedFolds(table, indexes, label_index)

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

    # print_double_line('Decision Tree Confusion Matrix Predictive Accuracy')

    # print_confusion(stratified_folds_matrix)



def main():
    table = remove_incomplete_rows(read_csv('auto-data.txt'))
    # table_titanic = remove_incomplete_rows(read_csv('titanic.txt')[1:])
    

    auto_decision_tree(table, [1, 4, 6], 0)
    

main()
