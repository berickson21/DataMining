from hw6 import remove_incomplete_rows, read_csv, get_column


class Apriori:

    def __init__(self, table, min_support, min_confidence):

        self.table = table
        self.length = float(len(self.table))
        self.min_support = min_support
        self.min_confidence = min_confidence

        self.att_domains = [set(get_column(self.table, att)) for att in range(len(self.table[0]))]
        self.all = [{item} for lists in self.att_domains for item in lists if self.get_support([item]) >= self.min_support]
        self.all_sets = []

        print self.all
       

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

    a = Apriori(table, 0.1, 0.80)


main()