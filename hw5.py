from math import log

from hw1 import get_column
from hw3 import read_csv, remove_incomplete_rows
from hw4_stratified_folds import StratifiedFolds, StratifiedFoldsTitanic


class DecisionTree:

    def __init__(self, training_set, att_indexes, label_index):
        self.training_set = training_set
        self.att_indexes = att_indexes
        self.label_index = label_index

    def calc_enew(instances, att_index, clas_index):
        
        D =  len(instances)
        freqs = att_freqs(instances, att_index, class_index)
        E_new = 0
        
        for att_val in freqs:
            D_j = freqs[att_val][1]
            probs - [(t / D_j) for (_, t) in freqs[att_val][0].items()]
        
        E_D_j = sum([p * log(p, 2) for p in probs])
        E_new += (D_j / D) * E_D_j

    def att_freqs(instances, att_index, class_index):
    
        att_vals = list(set(get_column(instances, att_index)))
        class_vals = list(set(get_column(instances, class_index)))

        result = {v: [{c:0 for c in class_vals}, 0] for v in att_vals}

        for row in instances:
            label = row[class_index]
            att_val = row[att_index]
            result[att_val][0][label] += 1
            result[att_val][1] += 1
        return result

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
