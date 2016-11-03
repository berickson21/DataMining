from hw5 import DecisionTree, AutoDecisionTree, TitanicDecisionTree, get_column, remove_incomplete_rows, read_csv
from copy import deepcopy
from random import shuffle, sample


class RandomForest:

    def __init__(self, training_set, att_indexes, label_index, m, n, f):

        self.training_set = training_set
        self.att_indexes = att_indexes
        self.label_indexes = label_index

        self.m = m  # select m best trees
        self.n = n  # number n trees
        self.f = f  # number of attribute indexes from att_indexes to build trees on

        self.initial_forest = [self.build_tree() for _ in range(self.n)]
        self.forest = None

        for i, tree in enumerate(self.initial_forest):
            # tree.save_graphviz_tree('trees/tree' + str(i))
            print tree.decision_tree

    def build_tree(self):

        bagging = self.bagging()
        remainder = bagging[0]
        test = bagging[1]

        tree = DecisionTree(remainder, sample(self.att_indexes, self.f), self.label_indexes)

        return tree

    def get_random_training_set(self):

        table = deepcopy(self.training_set)
        shuffle(table)

        return table[:(len(table)*2)/3]

    def classify(self, instance):

        classifications = [tree.classify(instance) for tree in self.forest]

        return classifications[0]

    def bagging(self):
        # make sure table is stratified
        table = deepcopy(self.training_set)
        shuffle(table)
        partition = (len(table) * 2) /3

        return table[0:partition], table[partition:]


def main():

    table = remove_incomplete_rows(read_csv('titanic.txt')[1:])
    d = RandomForest(table, [0, 1, 2], 3, 3, 6, 2)
    d = TitanicDecisionTree(table, [0, 1, 2], 3)

main()
