from hw6 import remove_incomplete_rows, read_csv, get_column, print_double_line
from copy import deepcopy
from operator import itemgetter
from tabulate import tabulate

import sys


class Apriori:

    def __init__(self, table, min_support, min_confidence, column_names):

        self.table = table
        self.column_names = column_names
        self.length = float(len(self.table))
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.num_att = len(table[0])
        self.att_domains = [list(set(get_column(self.table, att))) for att in range(len(self.table[0]))]
        self.candidate = [[item, i] for i in range(len(column_names)) for item in self.att_domains[i]]

        self.l1 = sorted([[item] for item in self.candidate
                          if self.get_support([item]) >= self.min_support], key=itemgetter(0))

        self.item_set = []
        self.generate_l_k(self.l1)

        self.rules = []
        self.create_rules()

        self.print_rules()

    def create_rules(self):

        for item in self.item_set:
            self.create_rule(item[0], [], item[1])

    def create_rule(self, left_side, right_side, support):

        if len(left_side) == 0:
            return

        right = deepcopy(right_side)
        items = left_side + right_side

        combinations = [[left_side[0:i] + left_side[i + 1:], right + [left_side[i]], support,
                         self.get_confidence(left_side[0:i] + left_side[i + 1:], items)] for i in range(len(left_side))]

        for row in combinations:
            if row[3] >= self.min_confidence and len(row[0]) > 0:
                self.create_rule(row[0], row[1], support)
                self.rules.append(row)

    def print_rules(self):
        rules = []

        for i, row in enumerate(self.rules):
            string = ''
            for item in row[0]:
                string += str(self.column_names[item[1]]) + '=' + item[0] + ' '

            string += '=> '

            for item in row[1]:
                string += str(self.column_names[item[1]]) + '=' + item[0] + ' '

            rules.append([i+1, string, row[2], row[3], 'lift'])

        headers = ['#', 'association rule', 'support', 'confidence', 'lift']

        print tabulate(rules, headers=headers, tablefmt="rst")

    # recursively creates grocery basket analysis
    def generate_l_k(self, initial_list):

        candidate = []

        for i, item in enumerate(initial_list):
            self.create_c_k(initial_list, candidate, initial_list[i], initial_list[i+1:])

        l_k = []

        for items in candidate:
            support = self.get_support(items)
            if support >= self.min_support:
                self.item_set.append([items, self.get_support(items)])
                l_k.append(items)

        if len(l_k) > 0:
            self.generate_l_k(l_k)

    # returns true if all k-1 subsets are in the initial list; otherwise, false.
    def check_k_minus_1(self, initial_list, items):
        return self.check_items_in_list(initial_list, sorted([items[0:i] + items[i+1:] for i in range(len(items))],
                                                             key=itemgetter(0)))

    # recursive helper function for check k_minus_1
    def check_items_in_list(self, initial_list, k_minus_1):

        if len(k_minus_1) == 0:
            return True
        else:
            return k_minus_1[0] in initial_list and self.check_items_in_list(initial_list, k_minus_1[1:])

    # returns candidate list (c_k) from the initial list
    def create_c_k(self, initial_list, candidate, left, right):

        for item in right:
            self.add_to_c_k(initial_list, candidate, left, item)

    # adds item to the candidate list if all k-1 subsets are supported in the initial_list
    def add_to_c_k(self, initial_list, candidate, left, right):

        if left[0:-1] == right[0:-1]:

            left_union_right = sorted([list(item) for item in set(tuple(item)
                                                  for item in sorted(list(left+right)))], key=itemgetter(0))

            if self.check_index(left_union_right, len(left)) and self.check_k_minus_1(initial_list, left_union_right):
                candidate.append(left_union_right)

    @staticmethod
    def check_index(item_set, num):
        return len(set([index for index in get_column(item_set, 1)])) == (num + 1)

    # returns true if all items in items are contained in item_set; otherwise, false.
    @staticmethod
    def contains(items, item_set):

        for item in items:
            if item_set[item[1]] != item[0]:
                return False

        return True

    # returns support count(left U right)/count(n)
    def get_support(self, items):
        return len([row for row in self.table if self.contains(items, row)])/self.length

    # returns confidence cont(left U right)/cunt(left)
    def get_confidence(self, left, items):
        return len([row for row in self.table if self.contains(items, row)]) / \
               float(len([row for row in self.table if self.contains(left, row)]))


def main():

    print_double_line('Titanic Association Rules')

    table = remove_incomplete_rows(read_csv('titanic.txt')[1:])
    a = Apriori(table, 0.20, 0.80, ['class', 'age', 'sex', 'survived'])

    print_double_line('Mushroom Association Rules')

    column_names = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
                    'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                    'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
                    'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']

    table = remove_incomplete_rows(read_csv('agaricus-lepiota.txt'))
    a = Apriori(table, 0.99, 0.80, column_names)

main()