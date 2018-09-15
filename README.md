# DMML Assignment 1 - Classification
#### Assignment by Arpan Biswas (BMC201604) and Satya Prakash Nayak (BMC201624)

*Here is the link to the output : [output.txt](https://drive.google.com/open?id=0B9NPosOUgkrNRTVNSWwyR0RjdWt5UFh5ajRXRVhSQ2VhRFVZ)*

## Libraries Used

* **numpy** : We used numpy.ndarray for storing data and sklearn packages also require numpy

* **pandas** : We used pandas.read_csv to read the input file with comma separated values and pandas.get_dummies for One-Hot encoding. We also made use of pandas.DataFrame which is the type of the value returned by pandas.read_csv. Later on we convert the data to numpy.ndarray.

* From **sklearn** : We used model_selection.KFold for performing the 10 Fold cross validation, tree.DecisionTreeClassifier for building the Decision Tree Classifier, naive_bayes.MultinomialNB for building Naïve Bayesian Classifier, svm.LinearSVC for building Linear SVM Classifier, metrics.confusion_matrix for calculating the confusion matrix and metrics.accuracy_score to calculate accuracy percentage

## Short Description of Code

In **main** method, **importData** is first called which returns *dat* and *target* where *dat* stores all the attributes of entire dataset after One-Hot encoding and *target* stores the class attribute. In **importData** we use pandas.read_csv to read the input file and perform One-Hot encoding of all the attributes in the data after separating out the class attribute.

Then we call the function **evaluateTenFold** for each of the 3 classifiers. In **evaluateTenFold**, we use sklearn.model_selection.KFold to perform the 10 fold cross validation and for each of 10 test and training dataset partition, we build our classifier using the training data and make predictions on the test data. Then we use sklearn.metrics.accuracy_score and sklearn.metrics.confusion_matrix to calculate and print accuracy percentage and confusion matrix respectively of our classifier. We then also calculate the average accuracy by taking average of all the accuracies we obtained in the 10 fold.

### Why use One-Hot encoding
One hot encoding is a process by which categorical variables are converted into a form that could be provided to ML algorithms to do a better job in prediction. We do not use simple numerical encoding because it might assume certain numerical features in the data that might give us wierd results.

## Time Taken and Memory Usage

The code takes a total of **9min 38.947s** to run, that is, it performs 10 fold cross validation for all 3 classifiers in that time.

We also measured separately the time taken to do 10 fold for each of the 3 classifiers by just commenting out the other 2 calls to evaluateTenFold for the other 2 classifiers in the main method. Here is the time taken for each:

* **Decision Tree Classifier** : 18.547s
* **Naïve Bayesian Classifier** : 5.166s
* **SVM Classifier** : 9min 11.019s

We also noted the memory usage upon execution of the code using Windows Task Manager. While doing 10-Fold Cross Validation on Decision Tree Classifier, RAM usage was around 130 MB, for Naïve Bayesian Classifier, it was around 100 MB and for SVM Classifier, it was around 200 MB.