from hw3 import remove_incomplete_rows, read_csv


class DecisionTree:

    def __init__(self, training_set, att_indexes, label_index):
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

    def check_labels(self, table):

        label = str(table[self.label_index])

        for row in table[1:]:
            if str(row[self.label_index]) != label:
                return False
        return True


class TitanicDecisionTree (DecisionTree):

    def __init__(self, training_set, att_indexes, label_index):
        DecisionTree.__init__(self, training_set, att_indexes, label_index)


class AutoDecisionTree (DecisionTree):

    def __init__(self, training_set, att_indexes, label_index):
        DecisionTree.__init__(self, training_set, att_indexes, label_index)


def main():
    table = remove_incomplete_rows(read_csv('auto-data.txt'))
    table_titanic = remove_incomplete_rows(read_csv('titanic.txt')[1:])

    d = DecisionTree(table, [1, 4, 6], 0)
    d.create_decision_tree()

main()
