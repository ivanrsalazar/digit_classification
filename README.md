# Digit Classification


### Overview
In this documentation, we present a comprehensive overview of our journey into the exciting world of machine learning through the analysis of the MNIST dataset. The primary objective of this project was to gain a solid understanding of the fundamental concepts and techniques in machine learning by developing classifiers for digit recognition. Throughout this endeavor, we explored various algorithms and methodologies, gradually building from simple binary classifiers to a more sophisticated multiclass classifiers.

### Binary Classifier
We kick-start our exploration by developing a binary least squares classifier. The objective here is to come up with some set of weights $W$, where given an input image, we can predict the correct label as much as possible. Which in turn means minimizing error as much as possible.

$$
\hat{y} = xW
$$

where $\hat{y}$ is the predicted output and $x$ is the corresponding input image, with error being defined as: $ e = y - \hat{y}$

We can then utilize the Least Squares equation in order to minimize error over a set of images, $X$

Least Squares:

$$
\min_{W} ||Y-XW|| ^2
$$

where $Y$ is the set of labels corresponding to the set of images $X$

To minimize the above equation, we would want to vary $\hat{W}$ such that $Y - X\hat{W}$ gets as close to 0 as possible. And to do that we must vary $\hat{W}$ such that:
 $$Y  ≈ \hat{Y} =  X\hat{W} $$
Next, by utilizing the psuedoinverse of the training data, we can come up with an optimized set of weights $\hat{W}$ that will successfully predict the output $y$

$$ 
X^TY = (X^TX)\hat{W}
$$

$$ 
(X^TX)^{-1}X^TY = \hat{W}  
$$



leading to:

$$ \hat{Y} =  X\hat{W}$$


 $$ \min ||Y - \hat{Y}||^2 $$

To implement the binary classifier for a specific digit, we perform a relabeling process on the dataset's labels. The goal is to assign a value of 1 to labels that correspond to the digit we are training for, and a value of -1 to labels that represent any other digit. This transformation enables us to create a clear distinction between the target digit and all other digits.

<p align="center">
  <img src="./photos/mnist_7.png" alt="Example Image">
</p>

For example, if we relabel all images said to be a 7 with a label of 1 and all other images with a label of -1, and solved for our set of weights minimizing error. The resulting binary classifer would be able to tell us if the above image is a 7 or not a 7.



### Multiclass Classifiers
Building upon the knowledge gained from binary classifiers, we extended our efforts to create two multiclass classifiers. By combining the outputs of multiple binary classifiers, we are able to create a One Vs. All and One Vs. One classifiers.

#### One Vs. All
For this multiclass classifier, we have 10 different classes and thus need to train 10 binary classifiers. Each binary classifier focuses on distinguishing one specific class from all the other classes. In order to arrive at the predicted output for the given input image, we feed the image to each of the 9 binary classifiers. 

Each classifier produces a score bounded between -1 and 1, indicating how likely the input belongs to its corresponding class. The final predicted class is the one associated with the binary classifier that returns the highest score.

#### One Vs. One
 To train all the classifiers using the One vs. One approach, we create binary classifiers for every possible pair of classes in the multi-class problem. Each binary classifier is responsible for distinguishing between the prediction of two specific classes. For instance, we need a classifier that can differentiate between predicting digit 3 or digit 5 given an input image. 

 We do this for a total of 45 binary classifiers, which happens to be $K(K-1)/2$

During the training phase, we will use the corresponding subset of the labeled training data for each pair of classes. For instance, for the classifier 3 vs. 5, we use only the images labeled as 3 or 5. For the classifier 1 vs. 9, we use only the images labeled as 1 or 9, and so on

When we want to classify a new input image, we pass it through all the binary classifiers. Each classifier makes a prediction (3 or 5, 1 or 9, etc.). The class that receives the most votes across all classifiers is then assigned as the final prediction for the new input image


### Randomized Feature Mapping

Next, instead of passing the training data directly through to train the above classifiers, we pass the set of images to be operated on by a random matrix $Z$ and random vector $b$, which are both drawn indepedently from a Normal Gaussian distribution

$$
Z = [z_1,z_2,...,z_L]^T
$$
where $z_1,z_2,...,z_L$ are the row vectors of $Z$
$$
b = [b_1,b_2,...,b_L]^T
$$
where $b_1,b_2,...,b_L$ are scalar values

$$
v_n(x,Z,b) = f(z_n^Tx + b_n)
$$

these values are then passed through a non-linear function, with those results used to train our new binary classifiers

Thus, our new set of feature vectors are equal to:
$$
    \begin{bmatrix}
    {v_1(x) = f(w_1^Tx + b_1)} \\
    {v_2(x)= f(w_2^Tx + b_2)} \\\
    {v_L(x) = f(w_L^Tx + b_L)} \\
    \end{bmatrix}
$$

#### Non Linear functions to use:
- Sigmoid: $f(x) = 1 / (1 + e^{-x})$
- Rectified Linear Unit (ReLU) : $f(x) = max(x,0)$














