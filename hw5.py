from hw3 import remove_incomplete_rows, read_csv


class DecisionTree:

    def __init__(self, training_set, att_indexes, label_index):
        self.training_set = training_set
        self.att_indexes = att_indexes
        self.label_index = label_index


class TitanicDecisionTree (DecisionTree):

    def __init__(self, training_set, att_indexes, label_index):
        DecisionTree.__init__(self, training_set, att_indexes, label_index)


class AutoDecisionTree (DecisionTree):

    def __init__(self, training_set, att_indexes, label_index):
        DecisionTree.__init__(self, training_set, att_indexes, label_index)


def main():
    table = remove_incomplete_rows(read_csv('auto-data.txt'))
    table_titanic = remove_incomplete_rows(read_csv('titanic.txt')[1:])

main()
