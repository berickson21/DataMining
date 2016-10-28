from math import log

from hw1 import get_column


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
