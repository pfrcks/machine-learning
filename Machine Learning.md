## ML Notes

#### Bias and Variance

##### Bias

bias of an estimator β^ is the difference between its expected value E[β^] and the true value of a parameter β being estimated.
More concretely, we compute the prediction bias as the difference between the expected prediction accuracy of our model and the true prediction accuracy.

##### Variance

The variance is simply the statistical variance of the estimator β^ and its expected value E[β^]

The variance is a measure of the variability of our model’s predictions if we repeat the learning process multiple times with small fluctuations in the training set.
#### Regularization

* L2 regularization or Ridge Regression cannot zero out coefficients.
* L1 regularization or Lasso Regression can zero out coefficients(good for sparse data).
* Elastic Net = Best of both worlds. (`lambda * ((j * L1)+((1-j)*L2)`)

#### Class Imbalance

* Get more data
* Generate synthetic data
* Undersampling/Oversampling
* Regularization

#### Classifier Types

Generative classifiers learn a model of the joint probability, p(x, y), of the inputs x and the label y, and make their predictions by using Bayes rules to calculate p(ylx), and then picking the most likely label y. Discriminative classifiers model the posterior p(ylx) directly, or learn a direct map from inputs x to the class labels.

#### Type I and II errors

* Type I error is false positive
* Type II error is false negative

#### Naive Bayes

What it essentially does it find the posterior probability given the prior. Called **Naive** because it assumes that X|Y is normally distributed with zero covariance.

#### Performance Metrics

* Accuracy sucks as a metric. [Misleading](https://en.wikipedia.org/wiki/Accuracy_paradox).
* Better to use Precision, Recall and ROC curves.
	* Precision : `TruePositives/TruePositives+FalsePositives`
	* Specificity: `TrueNegatives/TrueNegatives+FalsePositives`
	* Recall : `TruePositives/TruePositives+FalseNegatives`    = Sensitivity
Precision is the number of positive predictions divided by the total number of positive class values predicted. Recall is the number of positive predictions divided by the number of positive class values in the test data.
* ROC curves: Plot sensitivity vs 1-specificity. ROC stands for Receiver Operating Characteristic.
* F1 Score is `2*((precision*recall)/(precision+recall))`. Tells the balance between precision and recall.

#### Ensemble Techniques (Needs more work)
* Bagging : _Bootstrap AGGregatING_ uses a variation of data samples to train different base classifiers. Used with unstable classifiers which are sensitive to variations in training set(eg Decision Trees, Perceptrons). Used for reducing variance.
* Boosting : Used to convert weak learners to strong ones. Attaches weights proportionate to the amount of misclassification. Primarily used to reduce bias.
* Stacking : Uses an ensemble to extract features which are then again fed into an ensemble.

#### Evaluating ML Models

##### Accuracy

`print((TP + TN) / float(TP + TN + FP + FN))`

##### Sensitivity

When the actual value is positive, how often is the prediction correct? Also known as "True Positive Rate" or "Recall"
`sensitivity = TP / float(FN + TP)`

##### Specificity

When the actual value is negative, how often is the prediction correct?
`specificity = TN / (TN + FP)`

##### False Positive Rate

When the actual value is negative, how often is the prediction incorrect?
`false_positive_rate = FP / float(TN + FP)`

##### Precision

When a positive value is predicted, how often is the prediction correct?
`precision = TP / float(TP + FP)`

##### What should we use?

* Identify if FP or FN is more important to reduce
* Eg : Spam filter: optimize for precision or specificity
* Fraud Detector: optimize for sensitivity

When to use **ROC**

* insensitive to changes in class distribution (ROC curve does not
change if the proportion of positive and negative instances in the test
set are varied) 
* can identify optimal classification thresholds for tasks with differential
misclassification costs 

When to use ** PR **

* show the fraction of predictions that are false positives
* well suited for tasks with lots of negative instances


##### ROC Curver(Receiver Operating Curve)

Plots **sensitivity** vs **False Positive Rate**. The probabilistic interpretation of ROC-AUC score is that if you randomly choose a positive case and a negative case, the probability that the positive case outranks the negative case according to the classifier is given by the AUC

##### AUC

The percentage of the ROC plot that is underneath the curve. AUC is useful as a single number summary of classifier performance

Higher value = better classifier

AUC is useful even when there is high class imbalance (unlike classification accuracy)

#### Improving training/test

* Random resampling
* Stratified resampling
* Cross Validation

### Paper

DIVE(Distributional inclusion vector embedding) difference from skipgram

* all word embeddings and context embeddings are contrained to be non negative
* the weights of negative sampling for each word is inversely proportional to its frequency

DIVE was originally designed to perform unsupervised hypernymy task detection and its goal is to preserve the inclusion relation between two context features in the sparse bag of words

* When the co-occured context histogram of y includes that of word x, it means for all context words c in the vocalbulary v, c will co-occur more times with y than x.
* each basis index of DIVE corresponds to a topic and the embedding value at that index represents how often the word appears in the topic

