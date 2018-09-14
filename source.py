# DMML Assignment 1 by Arpan Biswas (BMC201604) and Satya Prakash Nayak (BMC201624)
# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

np.set_printoptions(threshold=np.nan)

# Import the Data
def importData():
    c4data = pd.read_csv('connect-4.data', sep= ',', header = None)
    # Seperate the target variable
    dat = pd.get_dummies(c4data.loc[:,:41]).values  # One Hot Encoding the attributes
    target = c4data.values[:,42]
    return dat, target

# Does a 10 fold cross validation with a given classifier
def evaluateTenFold(dat, target, clf):
    kf = KFold(n_splits=10)
    sumAcc = 0  # Used to store the sum of all 10 accuracies for the 10 fold
    for train_index, test_index in kf.split(dat):
        # Separate the train and test data
        dat_train, dat_test = dat[train_index], dat[test_index]
        target_train, target_test = target[train_index], target[test_index]
        # Train from data
        clf.fit(dat_train, target_train)
        # Make Predictions
        pred = clf.predict(dat_test)
        # Calculate Accuracy
        acc = accuracy_score(target_test, pred)*100
        sumAcc = sumAcc + acc
        # Print the Indices of current Test Data
        print('Indices of current Test Data:')
        print(test_index)
        # Print Predictions
        print('Predicted Values:')
        print(pred)
        # Print Accuracy
        print('Accuracy = ', acc, '%')
        # Print Confusion Matrix
        print('Confusion Matrix:')
        print(confusion_matrix(target_test, pred))
    # Print Average Accuracy
    print('Average Accuracy = ', sumAcc/10, '%')

# Main Method
def main():
    # Importing Data
    dat, target = importData()

    # Perform 10 fold cross validations for all 3 classifiers

    print('Performing 10 fold cross validation for Decision Tree Classifier...')
    evaluateTenFold(dat, target, DecisionTreeClassifier(criterion='entropy'))

    print('Performing 10 fold cross validation for Naive Bayesian Classifier...')
    evaluateTenFold(dat, target, MultinomialNB())

    print('Performing 10 fold cross validation for SVM Classifier...')
    evaluateTenFold(dat, target, LinearSVC())

     
# Calling main function
if __name__=="__main__":
    main()
