# Digit Classification


### Overview
In this documentation, we present a comprehensive overview of our journey into the exciting world of machine learning through the analysis of the MNIST dataset. The primary objective of this project was to gain a solid understanding of the fundamental concepts and techniques in machine learning by developing classifiers for digit recognition. Throughout this endeavor, we explored various algorithms and methodologies, gradually building from simple binary classifiers to a more sophisticated multiclass classifiers.

### Binary Classifier
We kick-started our exploration by developing a binary least squares classifier. The objective here is to come up with some set of weights $W$, where given an input image, we can successfully predict the output classification.

$$ \hat{y} = xW $$
where $\hat{y}$ is the predicted output and $x$ is the corresponding input image

Now, if we expand this to the entire dataset we will arrive to the Least Squares equation, where the goal is to minimize the error between the correct label y and predicted label $\hat{y}$

Least Squares:
$$
\min_{W} ||Y-XW|| ^2
$$

From here, we can arrive at:
 $$Y = XW$$
Next, by utilizing the psuedoinverse of the training data, we can come up with an optimized set of weights that will successfully predict the output of y:

$$ X^TY = (X^TX)W$$
$$ (X^TX)^{-1}X^TY = W  $$

To implement the binary classifier for a specific digit, we perform a relabeling process on the dataset's labels. The goal is to assign a value of 1 to labels that correspond to the digit we are training for, and a value of -1 to labels that represent any other digit. This transformation enables us to create a clear distinction between the target digit and all other digits.

<p align="center">
  <img src="./photos/mnist_7.png" alt="Example Image">
</p>

For example, if we relabel all images said to be a 7 with a label of 1 and all other images with a label of -1, and solved for our set of weights with the above equations. The resulting binary classifer would be able to tell us if the above image is a 7 or not a 7.



### Multiclass Classifiers
Building upon the knowledge gained from binary classifiers, we extended our efforts to create two multiclass classifiers. By combining the outputs of multiple binary classifiers, we were able to create a One Vs. All and One Vs. One classifiers.

#### One Vs. All
For this multiclass classifier, we have 10 different classes and thus need to train 10 binary classifiers. Each binary classifier focuses on distinguishing one specific class from all the other classes. In order to arrive at the predicted output for the given input image, we feed the image to each of the 9 binary classifiers. 

Each classifier produces a score bounded between -1 and 1, indicating how likely the input belongs to its corresponding class. The final predicted class is the one associated with the binary classifier that returns the highest score.

#### One Vs. One
For this classifier, the number of binary classifiers needed follows $K(K-1)/2$, in our case, this results to 45 different binary classifiers! Now thats a lot of classifiers!













