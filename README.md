# Linear Regression from Scratch

A clean, minimalist implementation of **Multiple Linear Regression** using NumPy. This project demonstrates the core mechanics of gradient descent, weight updates, and prediction without relying on high-level machine learning libraries like Scikit-Learn.

---

## 🚀 Features

* **Gradient Descent Optimization**: Uses batch gradient descent to minimize Mean Squared Error (MSE).
* **Vectorized Implementation**: Leverages NumPy matrix operations for efficient computation.
* **Scikit-Learn Style API**: Implements familiar `.fit()` and `.predict()` methods.

---

## 🛠️ Implementation Details

The model updates the weights ($w$) and bias ($b$) iteratively using the following gradient descent formulas:

$$\frac{\partial L}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$

$$\frac{\partial L}{\partial w} = \frac{1}{m} X^T (\hat{y} - y)$$

Where:
* $m$ is the number of samples.
* $\hat{y}$ is the predicted value ($Xw + b$).
* $y$ is the actual target value.

---

## 💻 Usage

### Prerequisites
Make sure you have NumPy installed:
```bash
pip install numpy