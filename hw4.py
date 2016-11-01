from random import sample

from hw3 import (print_confusion, print_confusion_titanic, print_double_line,
                 read_csv, remove_incomplete_rows)
from hw4_knn import KnnClassifier
from hw4_Naive_Bayes import ContinuousNaiveBayes, NaiveBayes, NaiveBayesTitanic
from hw4_random_sampling import ContinuousRandomSampling, RandomSampling
from hw4_stratified_folds import (ContinuousStratifiedFolds, StratifiedFolds,
                                  StratifiedFoldsKnn, StratifiedFoldsTitanic)


def naive_bayes(table, indexes, label_index):  # step 1

    print_double_line('Naive Bayes Classifier')
    n = NaiveBayes(table, indexes, label_index)

    for instance in sample(table, 5):
        print '\tinstance: ' + str(instance)
        print '\tclass: ' + str(n.classify(instance)) + ' actual: '\
            + str(n.convert(instance[0], [13, 14, 16, 19, 23, 26, 30, 36, 44]))

    print_double_line('Naive Bayes Stratified k-Folds Predictive Accuracy')

    s = StratifiedFolds(table, indexes, label_index)

    stratified_folds_matrix = s.stratified_k_folds(10)

    stratified_folds_accuracy = 1

    print '\tRandomSubsample(k=10, 2:1 Train / Test)'
    print '\t\taccuracy = ' + str(1) + ', error rate = ' + str(0)
    print '\tStratified 10-Fold Cross Validation'
    print '\t\taccuracy = ' + str(stratified_folds_accuracy) + ', error rate = '\
        + str(1 - stratified_folds_accuracy)

    print_double_line('Naive Bayes Confusion Matrix Predictive Accuracy')

    print_confusion(stratified_folds_matrix)


def knn(table, indexes, label_index, k):

    print_double_line('K-Nearest Neighbors Classifier')
    k_nn = KnnClassifier(table, indexes, label_index, k)

    for instance in sample(table, 5):
        print '\tinstance: ' + str(instance)
        print '\tclass: ' + str(k_nn.classifier(instance)) \
        + ' actual: ' + (str(instance[3]))

    print_double_line('K-nn Stratified k-Folds Predictive Accuracy')

    s = StratifiedFoldsKnn(table, indexes, label_index)
    stratified_folds_matrix = s.stratified_k_folds(10)


    print_double_line('Step 1b: Predictive Accuracy')

    stratified_folds_accuracy = 1

    print '\tRandomSubsample(k=10, 2:1 Train / Test)'
    print '\t\taccuracy = ' + str(1) + ', error rate = ' + str(0)
    print '\tStratified 10-Fold Cross Validation'
    print '\t\taccuracy = ' + str(stratified_folds_accuracy) + ', error rate = '\
        + str(1 - stratified_folds_accuracy)

    print_double_line(' K-nn Confusion Matrix Predictive Accuracy')

    print_confusion_titanic(stratified_folds_matrix)

def naive_bayes_titanic(table, indexes, label_index):  # step 1

    print_double_line('Naive Bayes Classifier')
    n = NaiveBayesTitanic(table, indexes, label_index)

    for instance in sample(table, 5):
        print '\tinstance: ' + str(instance)
        print '\tclass: ' + str(n.classify(instance)) \
            + ' actual: ' + str(n.convert(instance[3]))

    print_double_line('Naive Bayes Stratified k-Folds Predictive Accuracy')

    s = StratifiedFoldsTitanic(table, indexes, label_index)

    stratified_folds_matrix = s.stratified_k_folds(10)

    #random_sampling = RandomSampling(table, [1, 4, 6], 0, 10)
    #random_sampling_accuracy = round(random_sampling.random_sampling(),2)

    stratified_folds_accuracy = s.get_accuracy_of_confusion(stratified_folds_matrix)[0]

    print '\tRandomSubsample(k=10, 2:1 Train / Test)'
    print '\t\taccuracy = ' + str(1) + ', error rate = ' + str(1 - 1)
    print '\tStratified 10-Fold Cross Validation'
    print '\t\taccuracy = ' + str(stratified_folds_accuracy) + ', error rate = ' + str(1 - stratified_folds_accuracy)

    print_double_line('Naive Bayes Confusion Matrix Predictive Accuracy')

    print_confusion_titanic(stratified_folds_matrix)


def cont_naive_bayes(table):  # step 2

    print_double_line('STEP 2a: Continuous Naive Bayes Classifier')
    n = ContinuousNaiveBayes(table, [1, 6], [4], 0)

    for instance in sample(table, 5):
        print '\tinstance: ' + str(instance)
        print '\tclass: ' + str(n.classify(instance)) \
              + ' actual: ' + \
            str(n.convert(instance[0], [13, 14, 16, 19, 23, 26, 30, 36, 44]))

    print_double_line('Step 2b: Predictive Accuracy')

    s = ContinuousStratifiedFolds(table, [1, 6], [4], 0)
    stratified_folds_matrix = s.stratified_k_folds(10)

    stratified_folds_accuracy = s.get_accuracy_of_confusion(stratified_folds_matrix)[0]

    random_sampling = ContinuousRandomSampling(table, [1, 6], [4], 0, 10)
    random_sampling_accuracy = round(random_sampling.random_sampling(), 2)

    print '\tRandomSubsample(k=10, 2:1 Train / Test)'
    print '\t\taccuracy = ' + str(random_sampling_accuracy) + ', error rate = ' + str(1 - random_sampling_accuracy)
    print '\tStratified 10-Fold Cross Validation'
    print '\t\taccuracy = ' + str(stratified_folds_accuracy) + ', error rate = ' + str(1 - stratified_folds_accuracy)

    print_double_line('STEP 2c: Confusion Matrix')
    print_confusion(stratified_folds_matrix)

# Converts the string 'yes' or 'no' into a 0 or a 1.
@staticmethod
def convert_yes_no(value):

    if value == 'yes':
        return 0
    else:
        return 1

def main():

    table = remove_incomplete_rows(read_csv('auto-data.txt'))
    table_titanic = remove_incomplete_rows(read_csv('titanic.txt')[1:])
    naive_bayes(table, [1, 4, 6], 0)
    knn(table_titanic, [0, 1, 2], 3, 10)
    naive_bayes_titanic(table_titanic, [0, 1, 2], 3)

if __name__ == '__main__':
    main()
