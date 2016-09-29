from hw1 import read_csv, get_column, get_column_as_floats
from hw2 import remove_incomplete_rows

def distance(row, instance, n):
    
    return 1

def normalize(row):
    maximum = max(row)
    minimum = min(row)
    rng = maximum - minimum
    return[(r-minimum)/float(maximum) for r in row]


def select_class_label(top_k_rows):
    return 'Label'


def knn_classifier(trainingSet, n, instance, k):
    # trainingSet is a subset of the table
    # n is the number of at atributes
    # instance - trying to classify
    # size of comparision set

    distances = []

    for row in trainingSet:
        d = distance(row, instance, n)
        distances.append([d, row])

    distances.sort(key=lambda x: x[0])
    top_k_rows = distances[:k]

    label = select_class_label(top_k_rows)

    return label


def main():
    table = read_csv('auto-data.txt')
    table = remove_incomplete_rows(table)


if __name__ == '__main__':
    main()