# DMML Assignment 1 by Arpan Biswas (BMC201604) and Satya Prakash Nayak (BMC201624)
# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
 
# Import the Data
def importData():
    c4data = pd.read_csv('connect-4.data', sep= ',', header = None)
    # Seperate the target variable
    dat = pd.get_dummies(c4data.loc[:,:41]).values  # One Hot Encoding the attributes
    target = c4data.values[:,42]
    return dat, target

# Predict for Test Data and calculate accuracy
def testAccuracy(dat_test, target_test, clf):
    # Make Predictions
    pred = clf.predict(dat_test)
    # Print Accuracy
    print('Accuracy = ', accuracy_score(target_test, pred)*100, '%')

# Does a 10 fold cross validation with a given classifier
def evaluateTenFold(dat, target, clf):
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(dat):
        # Separate the train and test data
        dat_train, dat_test = dat[train_index], dat[test_index]
        target_train, target_test = target[train_index], target[test_index]
        # Train from data
        clf.fit(dat_train, target_train)
        # Print Accuracies
        testAccuracy(dat_test, target_test, clf)

# Main Method
def main():
    # Importing Data
    dat, target = importData()

    # Perform 10 fold cross validations for all 3 classifiers

    print('Performing 10 fold cross validation for a Decision Tree Classifier')
    evaluateTenFold(dat, target, DecisionTreeClassifier(criterion='entropy'))

    print('Performing 10 fold cross validation for a Naive Bayesian Classifier')
    evaluateTenFold(dat, target, MultinomialNB())

    print('Performing 10 fold cross validation for an SVM Classifier')
    evaluateTenFold(dat, target, LinearSVC())

     
# Calling main function
if __name__=="__main__":
    main()
