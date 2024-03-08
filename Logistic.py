# Import libraries
import numpy as np
from sklearn.datasets import load_iris

# Load iris data
iris = load_iris()
X = iris.data
y = iris.target

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def cost_function(X, y, theta):
  """Calculates the cost function (logistic loss)"""
  m = len(y)
  hypothesis = sigmoid(X.dot(theta))
  cost = -1/m * np.sum(y * np.log(hypothesis) + (1 - y) * np.log(1 - hypothesis))
  return cost

def gradient_descent(X, y, theta, learning_rate, num_iterations):
  """Performs gradient descent to update model parameters"""
  m = len(y)
  for i in range(num_iterations):
    prediction = sigmoid(X.dot(theta))
    error = prediction - y
    gradient = 1/m * X.T.dot(error)
    theta -= learning_rate * gradient
  return theta

# Initialize parameters
num_features = X.shape[1]
theta = np.zeros(num_features)

# Set hyperparameters
learning_rate = 0.01
num_iterations = 1000

# Train the model
theta = gradient_descent(X, y, theta, learning_rate, num_iterations)

# Make predictions using the trained model
def predict(X):
  hypothesis = sigmoid(X.dot(theta))
  return np.round(hypothesis).astype(int)

# Make predictions on new data (replace with your own data)
new_data = np.array([[5.1, 3.5, 1.4, 0.2]])
predicted_class = predict(new_data)

# Print prediction
print("Predicted class:", predicted_class[0])                                                                                                                               can u give simple codes for the same

