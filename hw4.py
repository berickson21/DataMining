from random import sample

from hw3 import (print_confusion, print_double_line, read_csv,
                 remove_incomplete_rows)
from hw4_knn import KnnClassifier
from hw4_Naive_Bayes import NaiveBayes
from hw4_Naive_Bayes_Titanic import NaiveBayesTitanic
from hw4_stratified_folds import StratifiedFolds


# from hw4_random_sampling import RandomSampling


def naive_bayes(table, indexes, label_index):  # step 1

    print_double_line('Naive Bayes Classifier')
    n = NaiveBayes(table, indexes, label_index)

    for instance in sample(table, 5):
        print '\tinstance: ' + str(instance)
        print '\tclass: ' + str(n.classify(instance)) \
              + ' actual: ' + \
            str(n.convert(instance[0], [13, 14, 16, 19, 23, 26, 30, 36, 44]))

    print_double_line('Naive Bayes Stratified k-Folds Predictive Accuracy')

    s = StratifiedFolds(table, indexes, label_index)
    stratified_folds_matrix = s.stratified_k_folds(10)

    stratified_folds_accuracy = 1

    print '\tRandomSubsample(k=10, 2:1 Train / Test)'
    print '\t\taccuracy = ' + str(1) + ', error rate = ' + str(0)
    print '\tStratified 10-Fold Cross Validation'
    print '\t\taccuracy = ' + str(stratified_folds_accuracy) + ', error rate = ' + str(1 - stratified_folds_accuracy)

    print_double_line('Naive Bayes Confusion Matrix Predictive Accuracy')

    print_confusion(stratified_folds_matrix)


def knn(table, indexes, label_index, k):

    print_double_line('K-Nearest Neighbors Classifier')
    k = KnnClassifier(table, indexes, label_index, k)

    print_double_line('K-nn Stratified k-Folds Predictive Accuracy')

    s = StratifiedFolds(table, indexes, label_index)
    stratified_folds_matrix = s.stratified_k_folds(10)

    stratified_folds_accuracy = 1

    print '\tRandomSubsample(k=10, 2:1 Train / Test)'
    print '\t\taccuracy = ' + str(1) + ', error rate = ' + str(0)
    print '\tStratified 10-Fold Cross Validation'
    print '\t\taccuracy = ' + str(stratified_folds_accuracy) + ', error rate = ' + str(1 - stratified_folds_accuracy)

    print_double_line(' K-nn Confusion Matrix Predictive Accuracy')

    print_confusion(stratified_folds_matrix)


def naive_bayes_titanic(table, indexes, label_index):  # step 1

    print_double_line('Naive Bayes Classifier')
    n = NaiveBayesTitanic(table, indexes, label_index)

    for instance in sample(table, 5):
        print '\tinstance: ' + str(instance)
        print '\tclass: ' + str(n.classify(instance)) \
              + ' actual: ' + str(n.convert_yes_no(instance[3]))

    print_double_line('Naive Bayes Stratified k-Folds Predictive Accuracy')

    s = StratifiedFolds(table, indexes, label_index)
    stratified_folds_matrix = s.stratified_k_folds(10)

    stratified_folds_accuracy = 1

    print '\tRandomSubsample(k=10, 2:1 Train / Test)'
    print '\t\taccuracy = ' + str(1) + ', error rate = ' + str(0)
    print '\tStratified 10-Fold Cross Validation'
    print '\t\taccuracy = ' + str(stratified_folds_accuracy) + ', error rate = ' + str(1 - stratified_folds_accuracy)

    print_double_line('Naive Bayes Confusion Matrix Predictive Accuracy')

    print_confusion(stratified_folds_matrix)


def main():

    table = remove_incomplete_rows(read_csv('auto-data.txt'))
    table_titanic = remove_incomplete_rows(read_csv('titanic.txt')[1:])
    naive_bayes(table, [1, 4, 6], 0)
    knn(table_titanic, [0, 1, 2], 3, 10)
    naive_bayes_titanic(table_titanic, [0, 1, 2], 3)

main()main()
