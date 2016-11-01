from copy import deepcopy
from math import sqrt

from scipy import stats


class KnnClassifier:

    def __init__(self, training_set, indexes, label_index, k):
        self.table = training_set
        self.training_set = deepcopy(training_set)
        self.indexes = indexes
        self.label_index = 3
        self.k = k

    def classifier(self, instance):

        distance = []
        for row in self.training_set:
            distance.append([self.dist(instance, row), row])

        self.normalize(distance)
        distance.sort(key=lambda x: x[0])

        top_k_neighbors = distance[0:self.k]
        return self.get_label(top_k_neighbors)

    def dist(self, instance, row):

        accumulator = 0

        for i in self.indexes:
            if instance[i] == row[i]:
                accumulator += 0
            else:
                accumulator += 1

        return sqrt(accumulator)

    def get_label(self, top_k_neighbors):
        for i in top_k_neighbors:
            self.convert(i)
        return stats.mode([top[1] for top in top_k_neighbors])[0][0]

    def convert(self, val):

        if val == 'yes':
            return 0
        else:
            return 1

    def normalize(self, list):
        column = [row[0] for row in list]
        max_val = max(column)
        # print 'max val: ' + str(max_val)
        min_val = min(column)
        # print 'min val: ' + str(min_val)

        maxmin = (max_val - min_val) * 1.0
        # print maxmin
        return [(row[0] - min_val) / (maxmin) for row in list]
