import hw4_Naive_Bayes


class NaiveBayesTitanic(hw4_Naive_Bayes.NaiveBayes):

    def categorize_instance(self, row):

        row[4] = self.convert_yes_no(row[4])

# Converts the string 'yes' or 'no' into a 0 or a 1.
    @staticmethod
    def convert_yes_no(value):

        if value == 'yes':
            return 0
        else:
            return 1