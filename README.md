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

## Deep Learning Practitioner's Guide
### Assorted Tips and Tricks

* Always shuffle
* Expand dataset. For images do-> add noise, whitening, drop pixels, rotate and color shift, blur
* Overfit a small subset of data initially
* Use dropout after large layers
* MAX > LRN pooling
* ReLU/PReLU > everything else
* Apply ReLU/PReLu after max pooling, saves computation
* Always use Batch Normalisation
* Zero mean and unit var.
	`>>> X -= np.mean(X, axis = 0) # zero-center`
	`>>> X /= np.std(X, axis = 0) # normalize`
* Use ensembling with smaller models
* Use Xavier init on FCN(not on CNN)
* If your input data has a spatial parameter try to go for CNN's end to end(read SqueezeNet)
* Decrease momentum with epochs
* Gradient Normalization
* LR -> ratio between the update norm and the weight norm ~ 10-3.

### Preprocessing

Another form pre-processing normalizes each dimension so that the min and max along the dimension is -1 and 1 respectively. It only makes sense to apply this pre-processing if you have a reason to believe that different input features have different scales (or units), but they should be of approximately equal importance to the learning algorithm. In case of images, the relative scales of pixels are already approximately equal (and in range from 0 to 255), so it is not strictly necessary to perform this additional pre-processing step.

### Correlation 

In theory, it can also be helpful to remove correlations between features by using PCA or ZCA whitening. However, in practice you may run into numerical stability issues since you will need to invert a matrix. So this is worth considering, but takes some more careful application. 

### Dropout

Dropout provides an easy way to improve performance. It’s trivial to implement and there’s little reason to not do it. Remember to tune the dropout probability, and to not forget to turn off Dropout and to multiply the weights by (namely by 1-dropout probability) at test time. Also, be sure to train the network for longer. Unlike normal training, where the validation error often starts increasing after prolonged training, dropout nets keep getting better and better the longer you train them. So be patient.

 During training, dropout can be interpreted as sampling a Neural Network within the full Neural Network, and only updating the parameters of the sampled network based on the input data. (However, the exponential number of possible sampled networks are not independent because they share the parameters.) During testing there is no dropout applied, with the interpretation of evaluating an averaged prediction across the exponentially-sized ensemble of all sub-networks (more about ensembles in the next section). I
### Variance Calibration

One problem with the above suggestion is that the distribution of the outputs from a randomly initialized neuron has a variance that grows with the number of inputs. It turns out that you can normalize the variance of each neuron's output to 1 by scaling its weight vector by the square root of its fan-in (i.e., its number of inputs), which is as follows:
`>>> w = np.random.randn(n) / sqrt(n) # calibrating the variances with 1/sqrt(n)`
This is for calibrating neurons without ReLU. For ReLU use He et al. initialization
`>>> w = np.random.randn(n) * sqrt(2.0/n) # current recommendation`

### During Training

* Filters and Pooling Size : it is important to employ a small filter (e.g., 3times 3) and small strides (e.g., 1) with zeros-padding, which not only reduces the number of parameters, but improves the accuracy rates of the whole deep network. Meanwhile, a special case mentioned above, i.e., 3times 3 filters with stride 1, could preserve the spatial size of images/feature maps. For the pooling layers, the common used pooling size is of 2times 2.
* Learning rate. In addition, as described in a blog by Ilya Sutskever [2], he recommended to divide the gradients by mini batch size. Thus, you should not always change the learning rates (LR), if you change the mini batch size. 
* ![ Fine tuning](/home/amol/Downloads/table.png  "Fine tuning")

### Activation Functions

* Sigmoid : Saturate or kill gradients
	* this (local) gradient will be multiplied to the gradient of this gate's output for the whole objective. Therefore, if the local gradient is very small, it will effectively “kill” the gradient and almost no signal will flow through the neuron to its weights and recursively to its data. Additionally, one must pay extra caution when initializing the weights of sigmoid neurons to prevent saturation. For example, if the initial weights are too large then most neurons would become saturated and the network will barely learn.
	* Sigmoid outputs are not zero-centered. This is undesirable since neurons in later layers of processing in a Neural Network (more on this soon) would be receiving data that is not zero-centered. This has implications on the dynamics during gradient descent, because if the data coming into a neuron is always positive (e.g., x>0 element wise in f=w^Tx+b), then the gradient on the weights w will during back-propagation become either all be positive, or all negative (depending on the gradient of the whole expression f). This could introduce undesirable zig-zagging dynamics in the gradient updates for the weights. However, notice that once these gradients are added up across a batch of data the final update for the weights can have variable signs, somewhat mitigating this issue. Therefore, this is an inconvenience but it has less severe consequences compared to the saturated activation problem above.

* tanh : saturates but is zero centered
* ReLU : simple function, does not saturate,  large gradient flowing through a ReLU neuron could cause the weights to update in such a way that the neuron will never activate on any datapoint again. If this happens, then the gradient flowing through the unit will forever be zero from that point on. That is, the ReLU units can irreversibly die during training since they can get knocked off the data manifold.
* Leaky ReLU : f(x)=alpha x if x less than 0 and f(x)=x if xgeq 0, where alpha is a small constant
* ReLU, Leaky ReLU, PReLU and RReLU. In these figures, for PReLU, alpha_i is learned and for Leaky ReLU alpha_i is fixed. For RReLU, alpha_{ji} is a random variable keeps sampling in a given range, and remains fixed in testing.

### Regularizations

* L2 : heavily penalizes peaky weights and preferres diffuse weight vectors
* L1
* Max Norm Constraint : this corresponds to performing the parameter update as normal, and then enforcing the constraint by clamping the weight vector vec{w} of every neuron to satisfy parallel vec{w} parallel_2 less than c
* Dropout : discussed above
