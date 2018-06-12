# Logistic-regression-with-and-without-regularization
ML is being applied on the images to predict whether the given image is dark or bright (0 or 1)
Initially, image processing has been applied on all the images to find out whether the avg value of the pixels is dark or bright. These results are stored in the actual[] array.
I have applied gradient descent algo in logistic regression in which random initial weights are taken. A dot product is taken between weight and feature vector (the image pixels stretched to 1d from 2d, in my case, with an added bias). Now the result is passed through a sigmoid function to obtain a value between 0 and 1 in form of probability. Error is calculated from these resultant probabilities and the actual values using log error function for logistic regression.
Our aim is reduce this error. So 20 generations of gradient descent are applied on all the images. The error rate finally being obtained is around 0.34.
After this, regularization is applied on the same algo.
Graphs of first 10 weights vs. all the 20 generations have been plotted to show how the weights change over iterations in gradient descent (with and without regualarization.
