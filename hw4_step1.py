from hw2 import read_csv, remove_incomplete_rows, get_column


def convert(value, cutoffs):

    for i, item in enumerate(cutoffs):
        if value < item:
            return i+1
        elif value > cutoffs[-1]:
            return len(cutoffs)+1


def categorize(table, index, cutoffs):

    new_table = table[:]

    for row in new_table:
        row[index] = convert(row[index], cutoffs)

    return table


def get_probabilities(table):
    return 1


def get_labels(table, index):

    return list(set(get_column(table, index)))


def main():

    table = remove_incomplete_rows(read_csv('auto-data.txt'))
    table = categorize(table, 4, [1999, 2499, 2999, 3499])

    print table[100]

    table = categorize(table, 0, [13, 14, 16, 19, 23, 26, 30, 36, 44])

    print table[100]


main()
