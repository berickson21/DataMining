from __future__ import print_function
import csv


def summary_statistics(table, keyList):
    print('Summary Stats:')
    print("No. of instances: " + str(len(table)))
    print("Duplicates: " + str(find_duplicates(table, keyList)))

    stats = [['============', '=====', '=====', '=======', '======', '====='],
             ['attribute', 'min', 'max', 'mid', 'avg', 'med'],
             ['============', '=====', '=====', '=======', '======', '====='],
             ['mpg', minimum(table, 0), maximum(table, 0), mid(table, 0), round(average(table, 0), -1),
              median(table, 0)],
             ['displacement', minimum(table, 2), maximum(table, 2), mid(table, 2), round(average(table, 2), -1),
              median(table, 2)],
             ['horsepower', minimum(table, 3), maximum(table, 3), mid(table, 3), round(average(table, 3), -1),
              median(table, 3)],
             ['weight', minimum(table, 4), maximum(table, 4), mid(table, 4), round(average(table, 4), -1),
              median(table, 4)],
             ['acceleration', minimum(table, 5), maximum(table, 5), mid(table, 5), round(average(table, 5), -1),
              median(table, 5)],
             ['MSRP', minimum(table, 9), maximum(table, 9), mid(table, 9), round(average(table, 9), -1),
              median(table, 9)],
             ['============', '=====', '=====', '=======', '======', '=====']]

    cols = zip(*stats)
    col_widths = [max(len(str(value)) + 2 for value in col) for col in cols]
    format = ''.join(['%%%ds' % width for width in col_widths])
    for row in stats:
        print(format % tuple(row))


def run(filename, keyList):
    table = read_csv(filename)

    print(filename)
    print_line()
    print("No. of instances: " + str(len(table)))
    print("Duplicates: " + str(find_duplicates(table, keyList)))
    print()
    print_line()


def read_csv(filename):
    the_file = open(filename, 'r')
    the_reader = csv.reader(the_file, dialect='excel')
    table = []

    for row in the_reader:
        if len(row) > 0:
            table.append(row)
    the_file.close()

    return table


def create_dictionary(filename, keyList):
    with open(filename, mode='r') as infile:
        reader = csv.reader(infile)
        dictionary = {create_key(rows, keyList): rows for rows in reader}
    return dictionary


def create_key(row, keyList):
    if len(keyList) == 1:
        return row[keyList[0]]
    else:
        return str(row[keyList[0]]) + create_key(row, keyList[1:])


def combine_tables(file1, keyList1, file2, keyList2, newFile):
    table1 = create_dictionary(file1, keyList1)
    table2 = create_dictionary(file2, keyList2)

    newTable = []

    for key in table1:
        if key in table2:

            newTable.append(table2[key] + [table1[key][2]])
            table2.pop(key, None)
        else:
            line = table1[key]

            newTable.append(['NA', 'NA', 'NA', 'NA', 'NA', 'NA'] + [line[1]] + ['NA'] + [line[0]] + [line[2]])

    for key in table2:
        newTable.append(table2[key] + ['NA'])

    with open(newFile, mode='w') as csvfile:
        writer = csv.writer(csvfile)
        [writer.writerow(row) for row in newTable]


def find_duplicates(table, keyList):
    duplicates = []

    i = 1
    for row1 in table:
        for row2 in table[i:]:
            if check_keys(row1, row2, keyList):
                duplicates.append(row1)
                duplicates.append(row2)

        i += 1

    return duplicates


def check_keys(row1, row2, keyList):
    if len(keyList) == 1:
        return row1[keyList[0]] == row2[keyList[0]]
    else:
        return row1[keyList[0]] == row2[keyList[0]] and check_keys(row1, row2, keyList[1:])


def remove_incomplete_rows(table, indexList=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    newTable = []

    for row in table:
        if check_NA(row, indexList):
            newTable.append(row)

    return newTable


def replace_data(table, indexList, key):
    newTable = []

    for row in table:
        if not check_NA(row, indexList):
            subset = get_subset(table, key, row[key])
            for index in indexList:
                if row[index] == 'NA':
                    row[index] = average(subset, index)
        newTable.append(row)

    summary_statistics(newTable, [8, 6])


def get_subset(table, keyIndex, keyValue):
    subset = []

    for row in table:
        if row[keyIndex] == keyValue:
            subset.append(row)

    return subset


def check_NA(row, indexList):
    if len(indexList) == 1:
        return row[indexList[0]] != 'NA'
    else:
        return row[indexList[0]] != 'NA' and check_NA(row, indexList[1:])


def get_column(table, index):
    values = []

    for rows in table:
        if 'NA' != rows[index]:
            values.append(rows[index])

    return values


def get_column_as_floats(table, index):
    vals = []

    for rows in table:
        if rows[index] != 'NA':
            vals.append(float(rows[index]))
    return vals


def minimum(table, index):
    return min(get_column_as_floats(table, index))


def maximum(table, index):
    return max(get_column_as_floats(table, index))


def mid(table, index):
    column = get_column_as_floats(table, index)
    return (max(column) + min(column)) / 2.0


def average(table, index):
    column = get_column_as_floats(table, index)
    if len(column) == 0:
        return 'NA'
    else:
        return average_column(column)


def average_column(column):
    return sum(column) / float(len(column))


def median(table, index):
    column = get_column_as_floats(table, index)
    column.sort()
    mid = len(column)
    if mid % 2 == 0:
        return average_column([column[mid / 2], column[(mid / 2) - 1]])
    else:
        return column[mid / 2]


def print_line():
    print("--------------------------------------------------")


def print_double_line():
    print("============  =====  =====  =======  ======  =====")


def clean(table):
    clean = []

    for row in table:
        if (row[0] == 'NA' and row[9] != 'NA'):
            clean.append(row)

    clean.sort(key=lambda x: x[8])

    subset = get_subset(table, 8, clean[1][8])

    for row in subset:
        print(row)


def main():
    print_line()
    run('auto-prices.txt', [0, 1])
    run('auto-mpg.txt', [8, 6])
    run('auto-prices-nodups.txt', [0, 1])
    run('auto-mpg-nodups.txt', [8, 6])

    combine_tables('auto-prices-nodups.txt', [0, 1], 'auto-mpg-nodups.txt', [8, 6], 'auto-data.txt')

    run('auto-data.txt', [6, 8])
    run('auto-prices-clean.txt', [0, 1])
    run('auto-mpg-clean.txt', [8, 6])

    table = read_csv('auto-data.txt')

    print_line()
    print('combined table (saved as auto-data.txt):')
    print_line()
    summary_statistics(table, [8, 6])

    print_line()
    print('combined table (rows w/ missing values removed):')
    print_line()

    newTable = remove_incomplete_rows(table, [0, 2, 3, 4, 5, 9])

    print_line()
    print('combined table (missing values replaced w/ corresponding average):')
    print_line()

    replace_data(table, [0, 2, 3, 4, 5, 9], 8)

    print_line()
    print('combined table (missing values replaced by yearly average):')
    print_line()
    replace_data(table, [0, 2, 3, 4, 5, 9], 6)


if __name__ == '__main__':
    main()