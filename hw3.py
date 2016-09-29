from hw1 import read_csv, get_column, get_column_as_floats
from hw2 import remove_incomplete_rows, regression_line, get_regression_lines, scatter_plot, COLUMN_NAMES


def distance(row, instance, n):

    return 1


def normalize(col):
    maximum = max(col)
    minimum = min(col)
    rng = maximum - minimum
    return[(item - minimum)/float(rng) for item in col]


def select_class_label(top_k_rows):
    return 'Label'


def knn_classifier(trainingSet, n, instance, k):
    # trainingSet is a subset of the table
    # n is the number of at atributes
    # instance - trying to classify
    # size of comparision set

    distances = []

    for row in trainingSet:
        distances.append([distance(row, instance, n), row])

    distances.sort(key=lambda x: x[0])
    top_k_rows = distances[:k]

    label = select_class_label(top_k_rows)

    return label


def linear_regression_classification(table, instance, xIndex, yIndex):

    reg = regression_line(table, xIndex, yIndex)
    slope = reg[0]
    intercept = reg[1]

    return (slope * instance[xIndex]) + intercept


def main():
    table = read_csv('auto-data.txt')
    table = remove_incomplete_rows(table)


if __name__ == '__main__':
    main()