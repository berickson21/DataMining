
from math import log

from hw1 import get_column
from hw3 import read_csv, remove_incomplete_rows


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

        self.training_set = self.categorize_table(training_set)
        self.training_set = training_set
        self.att_indexes = att_indexes
        self.label_index = label_index
        self.decision_tree = []

    def create_decision_tree(self):

        self.decision_tree = self.group_by(self.training_set, 6)

        for key in self.decision_tree:
            self.decision_tree[key] = self.group_by(self.decision_tree[key], 1)

        for key in self.decision_tree:
            for row in self.decision_tree[key]:
                print row

    @staticmethod
    def group_by(table, index):
        print len(table)

        dictionary = {}

        for row in table:

            if row[index] in dictionary:
                dictionary[row[index]].append(row)
            else:
                dictionary.update({row[index]: []})
                print 'added key:  ' + str(row[index])

        return dictionary

    def calc_enew(self, instances, att_index, class_index):
        
        D =  len(instances)
        freqs = self.att_freqs(instances, att_index, class_index)
        E_new = 0
        
        for att_val in freqs:
            D_j = freqs[att_val][1]
            probs = [(t / D_j) for (_, t) in freqs[att_val][0].items()]
        
        E_D_j = sum([p * log(p, 2) for p in probs])
        E_new += (D_j / D) * E_D_j

    def att_freqs(self, instances, att_index, class_index):
    
        att_vals = list(set(get_column(instances, att_index)))
        class_vals = list(set(get_column(instances, class_index)))

        result = {v: [{c:0 for c in class_vals}, 0] for v in att_vals}

        for row in instances:
            label = row[class_index]
            att_val = row[att_index]
            result[att_val][0][label] += 1
            result[att_val][1] += 1
        return result

    def same_class(self, table):
        # Returns true if all instances have same class value

        label = str(table[self.label_index])

        for row in table[1:]:
            if str(row[self.label_index]) != label:
                return False
        return True

    def partition_stats(self, instances, class_index):
        # List of stats: [[label1, occ1, total1], [label2, occ2, total2]...
        pass

    def partition_instances(self, instances, att_indexes, att_domains):
        # {att_val1: part1, att_val2:part2,...}
        pass

    def select_attribute(self, instances, att_indexes, class_index):
        # picks the attribute to partition on
        pass

    def tdit(self, instances, att_indexes, att_domains, class_index):
        # The main algorithm for the tree
        pass

    def classifier(self, decision_tree, instance):
        # returns label (really just navigating the tree given the instance)
        pass


class AutoDecisionTree (DecisionTree):

    def __init__(self, training_set, att_indexes, label_index):
        DecisionTree.__init__(self, training_set, att_indexes, label_index)
        Discretization.__init__(self)

    def categorize_instance(self, row):

        row[0] = self.convert(row[0], [13, 14, 16, 19, 23, 26, 30, 36, 44])
        row[4] = str(self.convert(row[4], [1999, 2499, 2999, 3499]))


class TitanicDecisionTree (DecisionTree):

    def __init__(self, training_set, att_indexes, label_index):
        DecisionTree.__init__(self, training_set, att_indexes, label_index)


def main():
    table = remove_incomplete_rows(read_csv('auto-data.txt'))
    table_titanic = remove_incomplete_rows(read_csv('titanic.txt')[1:])

    d = AutoDecisionTree(table, [1, 4, 6], 0)
    d.create_decision_tree()

main()
