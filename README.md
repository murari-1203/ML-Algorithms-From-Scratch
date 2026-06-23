# ML Algorithms From Scratch

A collection of Machine Learning algorithms implemented from scratch using **NumPy** and basic Python. The goal of this repository is to build a strong understanding of the mathematical foundations and inner workings of classical machine learning algorithms without relying on high-level libraries such as Scikit-Learn.

---

## 📈 Linear Regression

A clean, minimalist implementation of **Multiple Linear Regression** using NumPy. This implementation demonstrates the core mechanics of gradient descent, weight updates, and prediction.

### 🚀 Features

* Gradient Descent Optimization
* Vectorized NumPy Implementation
* Scikit-Learn Style API (`fit()` and `predict()`)

### 🛠️ Implementation Details

The model updates the weights ($w$) and bias ($b$) iteratively using the following gradient descent formulas:

$$
\frac{\partial L}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})
$$

$$
\frac{\partial L}{\partial w} = \frac{1}{m} X^T (\hat{y} - y)
$$

Where:

* $m$ is the number of samples.
* $\hat{y}$ is the predicted value ($Xw + b$).
* $y$ is the actual target value.

### 💻 Usage

```python
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

## 🎯 K-Nearest Neighbors (KNN)

A simple implementation of the **K-Nearest Neighbors** algorithm for classification using Euclidean Distance.

### 🚀 Features

* Distance-Based Classification
* Multi-Class Classification Support
* Majority Voting
* Configurable Number of Neighbors (K)

### 🛠️ Implementation Details

For a test point $x$, the Euclidean distance to every training sample is calculated as:

$$
d(x, x_i) = \sqrt{\sum_{j=1}^{n}(x_j - x_{ij})^2}
$$

The algorithm:

1. Computes the distance from the test point to all training samples.
2. Selects the K nearest neighbors.
3. Performs majority voting among the selected neighbors.
4. Assigns the most common class label.

### 💻 Usage

```python
knn = KNN(k=3)

knn.fit(X_train, y_train)

predictions = knn.predict(X_test)
```

---

## 🔜 Coming Soon

* Logistic Regression
* Decision Trees
* Random Forest
* Support Vector Machine (SVM)
* Naive Bayes
* K-Means Clustering
* Principal Component Analysis (PCA)
* Gradient Boosting

---

## 🛠️ Tech Stack

* Python
* NumPy
* Pandas
* Matplotlib
* Jupyter Notebook

---

## 🎯 Goal

* Understand machine learning algorithms from first principles.
* Strengthen mathematical intuition behind ML models.
* Build implementations without relying on Scikit-Learn.
* Create a practical reference for learning and interview preparation.

---

## ⭐ If you find this repository useful, consider giving it a star!
