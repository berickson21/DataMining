from hw6 import remove_incomplete_rows, read_csv

class Apriori:

    def __init__(self, table, min_support, min_confidence):

        self.table = table
        self.length = float(len(self.table))
        self.min_support = min_support
        self.min_confidence = min_confidence

    # returns true if all items in items are contained in item_set; otherwise, false.
    def contains(self, items, item_set):
        if len(items) == 1:
            return items[0] in item_set
        else:
            return items[0] in item_set and self.contains(items[1:], item_set)

    def get_support(self, items):

        return len([row for row in self.table if self.contains(items, row)])/self.length

    def get_confidence(self, left, items):
        return len([row for row in self.table if self.contains(items, row)])/\
               float(len([row for row in self.table if self.contains(left, row)]))


def main():
    table = remove_incomplete_rows(read_csv('titanic.txt')[1:])

    a = Apriori(table, 0.25, 0.80)

main()