# Deep Learning Notes

<!-- toc -->

- [Deep Learning Practitioner's Guide](#deep-learning-practitioners-guide)
    + [Assorted Tips and Tricks](#assorted-tips-and-tricks)
    + [Preprocessing](#preprocessing)
    + [Correlation](#correlation)
    + [Dropout](#dropout)
    + [Variance Calibration](#variance-calibration)
    + [During Training](#during-training)
    + [Activation Functions](#activation-functions)
    + [Regularizations](#regularizations)
- [Notes from CS231n](#notes-from-cs231n)
    + [Linear Classify](#linear-classify)
    + [Optimization - I](#optimization---i)
    + [Neural Networks - I](#neural-networks---i)
    + [Neural Networks - II](#neural-networks---ii)
      - [Preprocessing](#preprocessing-1)
      - [Weight Initialization](#weight-initialization)
      - [Regularization](#regularization)
      - [Loss functions](#loss-functions)
    + [Neural Networks - III](#neural-networks---iii)
      - [Parameter Updates](#parameter-updates)
      - [Adaptive Learning rates](#adaptive-learning-rates)
    + [Layers](#layers)
      - [Convolutional and Pooling Layer](#convolutional-and-pooling-layer)
      - [Batch Norm Layer](#batch-norm-layer)
      - [Upsampling Layer(Deconv/Transposed conv)](#upsampling-layerdeconvtransposed-conv)
      - [Skip connections](#skip-connections)
      - [Fully connected layer](#fully-connected-layer)
      - [Highway Layer](#highway-layer)
    + [Transfer Learning](#transfer-learning)
- [Popular architectures](#popular-architectures)
    + [AlexNet](#alexnet)
    + [VGGNet](#vggnet)
    + [GoogLeNet](#googlenet)
    + [ResNet](#resnet)
- [Object Detection](#object-detection)
    + [R-CNN(Region based CNN)](#r-cnnregion-based-cnn)
    + [Fast R-CNN](#fast-r-cnn)
    + [Faster R-CNN](#faster-r-cnn)
    + [YOLO](#yolo)
- [Semantic Segmentation](#semantic-segmentation)
    + [FCN](#fcn)
- [Classification](#classification)
- [Classification + Localization](#classification--localization)
- [Instance Segmentation](#instance-segmentation)

<!-- tocstop -->

## Deep Learning Practitioner's Guide

#### Assorted Tips and Tricks

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
* You want weight_updates / weight_magnitudes around 0.001, if high decrease LR
* Train ensembles, at test average their results
* Initialize ReLU with slightly positive bias to avoid dead neurons
* Use float32 or float16 instead of float64

#### Preprocessing

Another form pre-processing normalizes each dimension so that the min and max along the dimension is -1 and 1 respectively. It only makes sense to apply this pre-processing if you have a reason to believe that different input features have different scales (or units), but they should be of approximately equal importance to the learning algorithm. In case of images, the relative scales of pixels are already approximately equal (and in range from 0 to 255), so it is not strictly necessary to perform this additional pre-processing step.

#### Correlation

In theory, it can also be helpful to remove correlations between features by using PCA or ZCA whitening. However, in practice you may run into numerical stability issues since you will need to invert a matrix. So this is worth considering, but takes some more careful application.

#### Dropout

* Dropout provides an easy way to improve performance. It’s trivial to implement and there’s little reason to not do it. Remember to tune the dropout probability, and to not forget to turn off Dropout and to multiply the weights by (namely by 1-dropout probability) at test time. Also, be sure to train the network for longer. Unlike normal training, where the validation error often starts increasing after prolonged training, dropout nets keep getting better and better the longer you train them. So be patient.

* During training, dropout can be interpreted as sampling a Neural Network within the full Neural Network, and only updating the parameters of the sampled network based on the input data. (However, the exponential number of possible sampled networks are not independent because they share the parameters.) During testing there is no dropout applied, with the interpretation of evaluating an averaged prediction across the exponentially-sized ensemble of all sub-networks (more about ensembles in the next section).

*  Dropout makes the network unable to relay on a single feature. Thus, forcing it to generate redundant representation. In this way, you will be representing an object with redundant descriptors, detectors, etc. So in test time, if some features can not be detected, you still can really on the other ones.

* With p = 0.5 dropout prob, using all inputs in the forward pass would inflate the activations by 2x from what the network was ”used to” during training. Thus, you have to compensate by scaling the activations back down by 1/2. In this way, the ouput at test time is equal to the expected output at training time. Let’s see a naive implementation:

```python
	def train_step(X):
	# forward pass for example 3−layer neural network
	## Inverted dropout
		H1 = np.maximum(0, np.dot(W1, X) + b1)
		U1 = (np.random.rand(∗H1.shape) < p) / p # first dropout mask. Notice /p!
		H1 ∗= U1 # drop!
		H2 = np.maximum(0, np.dot(W2, H1) + b2)
		U2 = (np.random.rand(∗H2.shape) < p) / p # second dropout mask. Notice /p!
		H2 ∗= U2 # drop!
		out = np.dot(W3, H2) + b3
```

#### Variance Calibration

One problem with the above suggestion is that the distribution of the outputs from a randomly initialized neuron has a variance that grows with the number of inputs. It turns out that you can normalize the variance of each neuron's output to 1 by scaling its weight vector by the square root of its fan-in (i.e., its number of inputs), which is as follows:
`>>> w = np.random.randn(n) / sqrt(n) # calibrating the variances with 1/sqrt(n)`
This is for calibrating neurons without ReLU. For ReLU use He et al. initialization
`>>> w = np.random.randn(n) * sqrt(2.0/n) # current recommendation`

#### During Training

* Filters and Pooling Size : it is important to employ a small filter (e.g., 3times 3) and small strides (e.g., 1) with zeros-padding, which not only reduces the number of parameters, but improves the accuracy rates of the whole deep network. Meanwhile, a special case mentioned above, i.e., 3times 3 filters with stride 1, could preserve the spatial size of images/feature maps. For the pooling layers, the common used pooling size is of 2times 2.
* Learning rate. In addition, as described in a blog by Ilya Sutskever [2], he recommended to divide the gradients by mini batch size. Thus, you should not always change the learning rates (LR), if you change the mini batch size.
* ![ Fine tuning](/home/amol/Downloads/table.png  "Fine tuning")

#### Activation Functions

* Sigmoid : Saturate or kill gradients
	* this (S)(1-S) where S is the sigmoid func (local) gradient will be multiplied to the gradient of this gate's output for the whole objective. Therefore, if the local gradient is very small, it will effectively “kill” the gradient and almost no signal will flow through the neuron to its weights and recursively to its data. Additionally, one must pay extra caution when initializing the weights of sigmoid neurons to prevent saturation. For example, if the initial weights are too large then most neurons would become saturated and the network will barely learn.
	* Sigmoid outputs are not zero-centered. This is undesirable since neurons in later layers of processing in a Neural Network (more on this soon) would be receiving data that is not zero-centered. This has implications on the dynamics during gradient descent, because if the data coming into a neuron is always positive (e.g., x>0 element wise in f=w^Tx+b), then the gradient on the weights w will during back-propagation become either all be positive, or all negative (depending on the gradient of the whole expression f). This could introduce undesirable zig-zagging dynamics in the gradient updates for the weights. However, notice that once these gradients are added up across a batch of data the final update for the weights can have variable signs, somewhat mitigating this issue. Therefore, this is an inconvenience but it has less severe consequences compared to the saturated activation problem above.

* tanh : saturates but is zero centered
* ReLU : simple function, does not saturate,  large gradient flowing through a ReLU neuron could cause the weights to update in such a way that the neuron will never activate on any datapoint again. If this happens, then the gradient flowing through the unit will forever be zero from that point on. That is, the ReLU units can irreversibly die during training since they can get knocked off the data manifold.
* Leaky ReLU : f(x)=alpha x if x less than 0 and f(x)=x if xgeq 0, where alpha is a small constant
* ReLU, Leaky ReLU, PReLU and RReLU. In these figures, for PReLU, alpha_i is learned and for Leaky ReLU alpha_i is fixed. For RReLU, alpha_{ji} is a random variable keeps sampling in a given range, and remains fixed in testing.

#### Regularizations

* L2 : heavily penalizes peaky weights and preferres diffuse weight vectors
* L1
* Max Norm Constraint : this corresponds to performing the parameter update as normal, and then enforcing the constraint by clamping the weight vector vec{w} of every neuron to satisfy parallel vec{w} parallel_2 less than c
* Dropout : discussed above

## Notes from CS231n

#### Linear Classify

* Unlike kNN classifier, the advantage of this parametric approach is that once we learn the parameters we can discard the training data.
* Additionally, the prediction for a new test image is fast since it requires a single matrix multiplication with W, not an exhaustive comparison to every single training example.
* We introduced the bias trick, which allows us to fold the bias vector into the weight matrix for convenience of only having to keep track of one parameter matrix.
* We defined a loss function (we introduced two commonly used losses for linear classifiers: the SVM and the Softmax) that measures how compatible a given set of parameters is with respect to the ground truth labels in the training dataset. We also saw that the loss function was defined in such way that making good predictions on the training data is equivalent to having a small loss.

#### Optimization - I

* We saw that the gradient of a function gives the steepest ascent direction and we discussed a simple but inefficient way of computing it numerically using the finite difference approximation (the finite difference being the value of h used in computing the numerical gradient).
* We discussed the tradeoffs between computing the numerical and analytic gradient. The numerical gradient is simple but it is approximate and expensive to compute. The analytic gradient is exact, fast to compute but more error-prone since it requires the derivation of the gradient with math. Hence, in practice we always use the analytic gradient and then perform a gradient check, in which its implementation is compared to the numerical gradient.

#### Neural Networks - I

* Sigmoid : Prone to saturation and gradient killing. Sigmoid outputs are not zero centered.
* ReLU, Leaky ReLU: gradients might die
* maxout : The Maxout neuron computes the function  max(wT1x+b1,wT2x+b2)

#### Neural Networks - II

##### Preprocessing

* Mean subtraction : `X -= np.mean(X, axis = 0)`
* Normalization : `X /= np.std(X, axis = 0)`
* Whitening : `Xwhite = Xrot / np.sqrt(S + 1e-5)`. The whitening operation takes the data in the eigenbasis and divides every dimension by the eigenvalue to normalize the scale. The geometric interpretation of this transformation is that if the input data is a multivariable gaussian, then the whitened data will be a gaussian with zero mean and identity covariance matrix.
```python
X -= np.mean(X, axis = 0) # zero-center the data (important)
cov = np.dot(X.T, X) / X.shape[0] # get the data covariance matrix
U,S,V = np.linalg.svd(cov)
Xrot = np.dot(X, U) # decorrelate the data
Xrot_reduced = np.dot(X, U[:,:100])
```
* NOTE: An important point to make about the preprocessing is that any preprocessing statistics (e.g. the data mean) must only be computed on the training data, and then applied to the validation / test data

##### Weight Initialization

* All zero: because if every neuron in the network computes the same output, then they will also all compute the same gradients during backpropagation and undergo the exact same parameter updates.
* Small random numbers : he idea is that the neurons are all random and unique in the beginning, so they will compute distinct updates and integrate themselves as diverse parts of the full network. The implementation for one weight matrix might look like W = 0.01* np.random.randn(D,H), where randn samples from a zero mean, unit standard deviation gaussian.
* he distribution of the outputs from a randomly initialized neuron has a variance that grows with the number of inputs. the recommended heuristic is to initialize each neuron’s weight vector as: w = np.random.randn(n) / sqrt(n), where n is the number of its inputs. This ensures that all neurons in the network initially have approximately the same output distribution and empirically improves the rate of convergence.
* Var(w)=2/(nin+nout) : glorot
* W = np.random.randn(n) * sqrt(2.0/n) : He et al for ReLu
* Batch Normalization : applying this technique usually amounts to insert the BatchNorm layer immediately after fully connected layers (or convolutional layers, as we’ll soon see), and before non-linearities.

##### Regularization

* L1
* L2
* Max Norm
* Dropout : While training, dropout is implemented by only keeping a neuron active with some probability p (a hyperparameter), or setting it to zero otherwise. Test : `H1 = np.maximum(0, np.dot(W1, X) + b1) * p`.
* Inverted Dropout: to prevent test time extra computations.
```python
H1 = np.maximum(0, np.dot(W1, X) + b1)
U1 = (np.random.rand(*H1.shape) < p) / p # first dropout mask. Notice /p!
H1 *= U1
 #as compared to normal
H1 = np.maximum(0, np.dot(W1, X) + b1)
U1 = np.random.rand(*H1.shape) < p # first dropout mask
H1 *= U1
#and during test
H1 = np.maximum(0, np.dot(W1, X) + b1)
```

##### Loss functions

* SVM : Li=∑j≠yimax(0,fj−fyi+1)
* Softmax : Li=−log(efyi∑jefj)

#### Neural Networks - III

##### Parameter Updates

* SGD
* Momentum
```python
# Momentum update
v = mu * v - learning_rate * dx # integrate velocity
x += v # integrate position
```
* Nesterov Momentum: when the current parameter vector is at some position x, then looking at the momentum update above, we know that the momentum term alone (i.e. ignoring the second term with the gradient) is about to nudge the parameter vector by mu * v. Therefore, if we are about to compute the gradient, we can treat the future approximate position x + mu * v as a “lookahead” - this is a point in the vicinity of where we are soon going to end up. Hence, it makes sense to compute the gradient at x + mu * v instead of at the “old/stale” position x.
```python
x_ahead = x + mu * v
# evaluate dx_ahead (the gradient at x_ahead instead of at x)
v = mu * v - learning_rate * dx_ahead
x += v
```
* Annealing the learning rate: Step/Exponential/(1/t) decay

##### Adaptive Learning rates

* Adagrad
```python
# Assume the gradient dx and parameter vector x
cache += dx**2
x += - learning_rate * dx / (np.sqrt(cache) + eps)
```
* RMSProp
```python
cache = decay_rate * cache + (1 - decay_rate) * dx**2
x += - learning_rate * dx / (np.sqrt(cache) + eps)
```
* Adam
```python
m = beta1*m + (1-beta1)*dx
v = beta2*v + (1-beta2)*(dx**2)
x += - learning_rate * m / (np.sqrt(v) + eps)
```
* Search for hyperparameters on log scale. For example, a typical sampling of the learning rate would look as follows: learning_rate = 10 ** uniform(-6, 1)

#### Layers

##### Convolutional and Pooling Layer

* Conv -> ReLU -> Pool
* ![image](https://github.com/pfrcks/machine-learning/blob/master/cnn.png?raw=true)
* Dilation: This can be very useful in some settings to use in conjunction with 0-dilated filters because it allows you to merge spatial information across the inputs much more agressively with fewer layers. For example, if you stack two 3x3 CONV layers on top of each other then you can convince yourself that the neurons on the 2nd layer are a function of a 5x5 patch of the input (we would say that the effective receptive field of these neurons is 5x5). If we use dilated convolutions then this effective receptive field would grow much quicker. A 3x3 kernel with a dilation rate of 2 will have the same field of view as a 5x5 kernel, while only using 9 parameters
* ![image](https://github.com/pfrcks/machine-learning/blob/master/pool.png?raw=true)
* Conversely, any FC layer can be converted to a CONV layer. For example, an FC layer with K=4096 that is looking at some input volume of size 7×7×512 can be equivalently expressed as a CONV layer with F=7,P=0,S=1,K=4096. In other words, we are setting the filter size to be exactly the size of the input volume, and hence the output will simply be 1×1×4096 since only a single depth column “fits” across the input volume, giving identical result as the initial FC layer.
* A neuron on the second CONV layer has a 3x3 view of the first CONV layer, and hence by extension a 5x5 view of the input volume. Similarly, a neuron on the third CONV layer has a 3x3 view of the 2nd CONV layer, and hence a 7x7 view of the input volume. Suppose that instead of these three layers of 3x3 CONV, we only wanted to use a single CONV layer with 7x7 receptive fields. These neurons would have a receptive field size of the input volume that is identical in spatial extent (7x7), but with several disadvantages. First, the neurons would be computing a linear function over the input, while the three stacks of CONV layers contain non-linearities that make their features more expressive. Second, if we suppose that all the volumes have C channels, then it can be seen that the single 7x7 CONV layer would contain C×(7×7×C)=49C2 parameters, while the three 3x3 CONV layers would only contain 3×(C×(3×3×C))=27C2 parameters.
* zero padding `= (k-1)/2` where k is filter size
* output size `= (w-k+2P)/2 +1` where w is input size, p is padding and s is stride
* No. of FLOPS : input depth x output depth x o/p_feature_map_width x o/p_feature_map_height x kernel_width x kernel_height
* Pooling layers help prevent overfitting and reduce computation
* CNN gives location invariance while Pooling gives translation, rotation and scaling invariance
* The output volume depth is equal to the number of filters
* In the output volume, the d-th depth slice (of size W2 × H2 is the result of performing a valid convolution of the d-th filter over the input volume with a stride of S, and then offset by d-th bias.
* Normally, in the same conv layer, all filters have the same dimensions so that special optimized routines can be invoked.

##### Batch Norm Layer

Batch normalization potentially helps in two ways: faster learning and higher overall accuracy. The improved method also allows you to use a higher learning rate, potentially providing another boost in speed.

As the data flows through a deep network, the weights and parameters adjust those values, sometimes making the data too big or too small again - a problem the authors refer to as ”internal covariate shift”. By normalizing the data in each mini-batch, this problem is largely avoided. Adding batch normalization normally slows 30%.

ut the problem appears in the intermediate layers because the distribution of the activations is constantly changing during training. This slows down the training process because each layer must learn to adapt themselves to a new distribution in every training step. This problem is known as internal covariate shift.<Paste>

Covariate shift refers to the change in the input distribution to a learning system. In the case of deep networks, the input to each layer is affected by parameters in all the input layers. So even small changes to the network get amplified down the network. This leads to change in the input distribution to internal layers of the deep network and is known as internal covariate shift.

So what you usually have are fully connected or convoluational layers followed by batch normal- ization layer before the non-linearity. So they ensure that everything is roughly unit Gaussian at each step of the neural net. One problem is that its not clear that tanh wants exactly unit Gaussian. Because you what tanh to be able to make its outputs more or less defused (more or less saturated) so right now it would not be able to do that.

To solve this you do not only normalize X but you also allow the network to shift by gamma and add B. So after you have centred your data you are allowing the network through the backprop to shift and scale the distribution. Also note that the network can learn to undo this layer (it can learn to have the batch normalization layer to be an identity)

It acts somehow as a way of regularization because with batch norm when you have some kind of input x and it goes through the network its representation in some layer of the network is basically not only function of it but also whatever other examples are with x in the batch. Normally all examples are processed independently in parallel, but batch norm tides them together and so your representation at some layer is a function on whatever batch you happen to be sampled in at what it does is to jigger your place in the representation space in that layer. Which is a nice regularizer effect

Consequently, batch normalization adds two trainable parameters to each layer, so the normalized output is multiplied by a “standard deviation” parameter (gamma) and add a “mean” parameter (beta). In other words, batch normalization lets SGD do the denormalization by changing only these two weights for each activation, instead of losing the stability of the network by changing all the weights.

During testing, a single mean is used(calculated using running average of mean during train time)

##### Upsampling Layer(Deconv/Transposed conv)

Transposed convolutions – also called fractionally strided convolutions – work by swapping the forward and backward passes of a convolution. One way to put it is to note that the kernel defines a convolution, but whether its a direct convolution or a transposed convolution is determined by how the forward and backward passes are computed.
The transposed convolution is implemented as the backward pass of a corresponding non- transposed convolution. It can be thought of as dilating the input (by adding “stride - 1” zeros between adjacent input elements), padding it with the needed number of zeros so it is not out. And then, apply the convolution with the filter flipped 180 degrees.

Deconvolution layer is a very unfortunate name and should rather be called a transposed convolutional layer.

Visually, for a transposed convolution with stride one and no padding, we just pad the original input (blue entries) with zeroes (white entries) (Figure 1).

##### Skip connections

There was some information that was captured in the initial layers and was required for reconstruction during the up-sampling done using the FCN layer. If we would not have used the skip architecture that information would have been lost (or should say would have turned too abstract for it to be used further ). So the information that we had in the primary layers can be fed explicitly to the later layers using the skip architecture.

##### Fully connected layer


FC layers, also called affine layers, produce the high-level reasoning in the DNN. Neurons in a fully connected layer have full connections to all activations in the previous layer, as in regular Neural Networks. Their activations can hence be computed with a matrix multiplication followed by a bias offset.

##### Highway Layer

A Highway Layer is a type of Neural Network layer that uses a gating mechanism to control the information flow through a layer. Stacking multiple Highway Layers allows for training of very deep networks. Highway Layers work by learning a gating function that chooses which parts of the inputs to pass through and which parts to pass through a transformation function, such as a standard affine layer for example. The basic formulation of a Highway Layer is T * h(x) + (1 - T) * x, where T is the learned gating function with values between 0 and 1, h(x) is an arbitrary input transformation and x is the input. Note that all of these must have the same size.

#### Transfer Learning

* New dataset is small and similar to original dataset. Since the data is small, it is not a good idea to fine-tune the ConvNet due to overfitting concerns. Since the data is similar to the original data, we expect higher-level features in the ConvNet to be relevant to this dataset as well. Hence, the best idea might be to train a linear classifier on the CNN codes.
* New dataset is large and similar to the original dataset. Since we have more data, we can have more confidence that we won’t overfit if we were to try to fine-tune through the full network.
* New dataset is small but very different from the original dataset. Since the data is small, it is likely best to only train a linear classifier. Since the dataset is very different, it might not be best to train the classifier form the top of the network, which contains more dataset-specific features. Instead, it might work better to train the SVM classifier from activations somewhere earlier in the network.
* New dataset is large and very different from the original dataset. Since the dataset is very large, we may expect that we can afford to train a ConvNet from scratch. However, in practice it is very often still beneficial to initialize with weights from a pretrained model. In this case, we would have enough data and confidence to fine-tune through the entire network.


## Popular architectures

#### AlexNet

* Used ReLU
* Stacked conv layers
* Data Aug :  image translations, horizontal reflections, and patch extractions.
* Implement dropout

#### VGGNet

* two 3x3 conv layers has an effective receptive field of 5x5.
* 3 conv layers back to back have an effective receptive field of 7x7.
* As the spatial size of the input volumes at each layer decrease (result of the conv and pool layers), the depth of the volumes increase due to the increased number of filters as you go down the network.
* Interesting to notice that the number of filters doubles after each maxpool layer. This reinforces the idea of shrinking spatial dimensions, but growing depth.

#### GoogLeNet

*  one of the first CNN architectures that really strayed from the general approach of simply stacking conv and pooling layers on top of each other in a sequential structure.
*   9 Inception modules in the whole architecture,
*   No use of fully connected layers! They use an average pool instead, to go from a 7x7x1024 volume to a 1x1x1024 volume. This saves a huge number of parameters.

#### ResNet

Let H(x) be a function that you desire to obtian. In a typical net you would compute a squence of steps ReLu(ReLu(xw1+b1)*w2+b2) to transform x to H(x). Instead, in a ResNet you compute a delta to be added to the original input to obtain H(x).
What is nice about it is that in plain nets, gradients must flow through all the transforma- tions. Instead, in residual nets because it is addition (distributes the gradient equally to all its children), the gradient with flow through the (weights, ReLU) but will also skip this transformations and will go directly to the previous part and flow directly to the previous block. So the gradients can skip all the transformations and go directly to the first layer. In this way, you can train very fast the first layer which is doing simple statistics, and the rest of layers will learn to add to the single in between to make it work at the end.

## Object Detection

#### R-CNN(Region based CNN)

We use selective search to extract just 2000 regions from the image and he called them region proposals. First run a region proposal algorithm to obtain regions of interets. Then, warp this regions into a fixed size and run the ConvNet with regression head and classification head. The regression head objective is to output an offset to correct ”slightly wrong” region proposals.
Issues with RCNN

* Slow at test time
* SVM and regressor are post hoc
* selective search algorithm is a fixed algorithm. Therefore, no learning is happening at that stage

#### Fast R-CNN

Swap the order of extracting ROI and features. The input image is processed by Conv and ROIs(Project region proposal onto conv feature map) are proposed on it. Since FC layers expect fix size, we divide each ROI into a h*w size grid and do max pooling in each grid cell

#### Faster R-CNN

To solve the problem of region proposals algorithm botleneck in Fast R-CNN, Faster R-CNN proposes to extract the regions of interest with another network using the information of the last layer of the CNN net.

Use N anchor boxes at each location. Anchors are translation invariant: use the same ones at every location. For all the feature map points, at the original input image apply all anchors to the point corresponding to the current feature map point. So we take as features the convolutional feature map region corresponding to the anchor region in the original image.
For each of this anchor boxes it produces a score weather there is an object or not; and a finer localization with reference to the anchor (they are an offset to the anchor boxes)
So you are learning different weights for each anchor.
One network, four losses
* RPN classification (anchor good / bad)
* RPN regression (anchor to proposal)
* Fast R-CNN classification (over classes)
* Fast R-CNN regression (proposal to box)

#### YOLO

The idea is to solve detection as only a regression problem. Direct prediction using a CNN. Divide image into S × S grid, they use S = 7. The, within each grid cell predict
* B boxes: 4 coordinates + confidence, they use B = 2
* Class scores: C numbers
Regression from input image to output S × S × (5 ∗ B + C) tensor.
It can go at real-time at the expense of lower mAP than Faster R-CNN.

How YOLO works is that we take an image and split it into an SxS grid, within each of the grid we take m bounding boxes. For each of the bounding box, the network outputs a class probability and offset values for the bounding box. The bounding boxes having the class probability above a threshold value is selected and used to locate the object within the image.

## Semantic Segmentation

Segmentation task is different from classification task because it requires predicting a class for each pixel of the input image, instead of only 1 class for the whole input. Classification needs to understand what is in the input (namely, the context). However, in order to predict what is in the input for each pixel, segmentation needs to recover not only what is in the input, but also where.

Three common strategies are used:

* Multiscale
* Refinement
* Upsampling

#### FCN

Fully Convolutional Networks (FCNs) owe their name to their architecture, which is built only from locally connected layers, such as convolution, pooling and upsampling. Note that no dense layer is used in this kind of architecture. This reduces the number of parameters and computation time. Also, the network can work regardless of the original image size, without requiring any fixed number of units at any stage, givent that all connections are local. To obtain a segmentation map (output), segmentation networks usually have 2 parts :

* Downsampling path : capture semantic/contextual information
* Upsampling path : recover spatial information

The downsampling path is used to extract and interpret the context (what), while the upsampling path is used to enable precise localization (where). Furthermore, to fully recover the fine-grained spatial information lost in the pooling or downsampling layers, we often use skip connections.

A skip connection is a connection that bypasses at least one layer. Here, it is often used to transfer local information by concatenating or summing feature maps from the downsampling path with feature maps from the upsampling path. Merging features from various resolution levels helps combining context information with spatial information.

Metrics: Per pixel accuracy, Intersection over union

## Classification

Train a classification model with softmax loss. The input is the entire image and the output are C probabilities (one per class) of being in the image.

## Classification + Localization

*  Train (or download) a classification model (AlexNet, VGG, GoogLeNet)
* Attach a new fully-connected ”regression head” to the network to compute bounding boxes (x,y,w,h)
* Train the regression head only with SGD and L2 loss with the bounding boxes as ground-truth.You can backpropagate only till the regression head or the entire network. 2nd option will improve a little bit the accuracy at a expense of higher training computation cost. If you choose option 2 you will be changing the original Conv layers on which the classification head is trained. So there are two options. Or you have two independent networks: the original one (Conv+classification head) and the other one (Modified Conv+regression head). Or you train both at the same time so you will have only one model(classification head + regression head).
* Finally:
    - Classification head: Output are C numbers(one per class)
    - Regression head: Class agnostic gives 4 numbers(BB), class specific gives C*4

## Instance Segmentation

Detect instance, generate mask. Similar pipelines to object detection.

1. External segment proposals that outputs pixels not boxes
2. Produce a BBox of the segmented region
3. Take the BBox image an run it through a CNN
4. Take the BBox image and set the non segment proposal pixels to the mean image value of the dataset an run it through a CNN
5. Concatenate booth features an run it through a classifier
6. Refine the proposed region

Very similar to RCNN. Another FRCNN type which From the high resolution feature map propose regions, then reshape boxes and finally mask background and predict object class.
