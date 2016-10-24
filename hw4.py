from hw3 import print_confusion, print_double_line, remove_incomplete_rows, read_csv
from hw4_Naive_Bayes import NaiveBayes, ContinuousNaiveBayes
from hw4_stratified_folds import StratifiedFolds, ContinuousStratifiedFolds
# from hw4_random_sampling import RandomSampling

from random import sample


def naive_bayes(table):  # step 1

    print_double_line('STEP 1a: Naive Bayes Classifier')
    n = NaiveBayes(table, [1, 4, 6], 0)

    for instance in sample(table, 5):
        print '\tinstance: ' + str(instance)
        print '\tclass: ' + str(n.classify(instance)) \
              + ' actual: ' + str(n.convert(instance[0], [13, 14, 16, 19, 23, 26, 30, 36, 44]))

    print_double_line('Step 1b: Predictive Accuracy')

    s = StratifiedFolds(table, [1, 4, 6], 0)
    stratified_folds_matrix = s.stratified_k_folds(10)

    stratified_folds_accuracy = s.get_accuracy_of_confusion(stratified_folds_matrix)[0]
    print stratified_folds_accuracy

    print '\tRandomSubsample(k=10, 2:1 Train / Test)'
    print '\t\taccuracy = ' + str(1) + ', error rate = ' + str(0)
    print '\tStratified 10-Fold Cross Validation'
    print '\t\taccuracy = ' + str(stratified_folds_accuracy) + ', error rate = ' + str(1-stratified_folds_accuracy)

    print_double_line('STEP 1c: Confusion')

    print_confusion(stratified_folds_matrix)


def cont_naive_bayes(table):  # step 2

    print_double_line('STEP 2a: Continuous Naive Bayes Classifier')
    n = ContinuousNaiveBayes(table, [1, 6], [4], 0)

    for instance in sample(table, 5):
        print '\tinstance: ' + str(instance)
        print '\tclass: ' + str(n.classify(instance)) \
              + ' actual: ' + str(n.convert(instance[0], [13, 14, 16, 19, 23, 26, 30, 36, 44]))

    print_double_line('Step 2b: Predictive Accuracy')

    s = ContinuousStratifiedFolds(table, [1, 6], [4], 0)
    stratified_folds_matrix = s.stratified_k_folds(10)

    stratified_folds_accuracy = s.get_accuracy_of_confusion(stratified_folds_matrix)[0]

    print '\tRandomSubsample(k=10, 2:1 Train / Test)'
    print '\t\taccuracy = ' + str(1) + ', error rate = ' + str(0)
    print '\tStratified 10-Fold Cross Validation'
    print '\t\taccuracy = ' + str(stratified_folds_accuracy) + ', error rate = ' + str(1-stratified_folds_accuracy)

    print_double_line('STEP 2c: Confusion Matrix')

    print_confusion(stratified_folds_matrix)


def main():

    table = remove_incomplete_rows(read_csv('auto-data.txt'))
    naive_bayes(table)
    cont_naive_bayes(table)

main()
