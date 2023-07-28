# Digit Classification


### Overview
In this documentation, we present a comprehensive overview of our journey into the exciting world of machine learning through the analysis of the MNIST dataset. The primary objective of this project was to gain a solid understanding of the fundamental concepts and techniques in machine learning by developing classifiers for digit recognition. Throughout this endeavor, we explored various algorithms and methodologies, gradually building from simple binary classifiers to a more sophisticated multiclass classifiers.

### Binary Classifiers
We kick-started our exploration by developing binary classifiers using Least Squares. The first type can distinguish between a specific digit and all other digits, for example is this digit a 1 or not a 1?

<p align="center">
  <img src="./photos/mnist_7.png" alt="Example Image">
</p>


The next classifier would distinguish between a specific digit and another specific digit, for example is this digit a 3 or a 5? These two types of classifiers are known as One Vs All and One Vs One respectively.

<p align="center">
  <img src="./photos/mnist_5.png" alt="Example Image">
</p>


### Multiclass Classifiers
Building upon the knowledge gained from binary classifiers, we extended our efforts to create a multiclass classifier. By combining the outputs of multiple binary classifiers, we achieved the ability to recognize multiple digits simultaneously. Through rigorous experimentation and tuning, we achieved an impressive error rate of just 3%.












####