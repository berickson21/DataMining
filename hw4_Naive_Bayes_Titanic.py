import hw4_Naive_Bayes


class NaiveBayesTitanic(hw4_Naive_Bayes.NaiveBayes):

    def categorize_instance(self, row):

        row[4] = self.convert_yes_no(row[4])
