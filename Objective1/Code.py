import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Perceptron class definition
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.rand(input_size + 1)  # Initialize weights with random values
        self.loss_history = []

    def activation(self, x):
        # Activation function: returns 1 if x >= 0, else 0 (step function)
        return 1 if x >= 0 else 0

    def predict(self, x):
        # Insert bias input (1) and perform prediction
        x = np.insert(x, 0, 1)
        return self.activation(np.dot(self.weights, x))

    def train(self, X, y):
        # Training the perceptron using the provided input X and labels y
        for _ in range(self.epochs):
            total_error = 0
            for i in range(len(X)):
                # Insert bias input (1) and perform prediction
                x_i = np.insert(X[i], 0, 1)
                y_pred = self.activation(np.dot(self.weights, x_i))
                error = y[i] - y_pred  # Calculate error
                total_error += abs(error)  # Add the absolute error
                self.weights += self.learning_rate * error * x_i  # Update the weights
            self.loss_history.append(total_error)  # Track total error for each epoch

    def evaluate(self, X, y):
        # Evaluate the model by calculating accuracy
        correct = sum(self.predict(x) == y_i for x, y_i in zip(X, y))
        return correct / len(y)

    def plot_loss_curve(self):
        # Plot loss curve (error vs epochs)
        plt.plot(self.loss_history)
        plt.xlabel('Epochs')
        plt.ylabel('Total Error')
        plt.title('Loss Curve')
        plt.show()

    def get_confusion_matrix(self, X, y):
        # Get and display confusion matrix
        y_pred = [self.predict(x) for x in X]
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()


# NAND truth table
X_nand = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_nand = np.array([1, 1, 1, 0])

# XOR truth table
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# Train Perceptron for NAND
t_nand = Perceptron(input_size=2)
t_nand.train(X_nand, y_nand)
nand_accuracy = t_nand.evaluate(X_nand, y_nand)
print("NAND Perceptron Accuracy:", nand_accuracy)
t_nand.plot_loss_curve()
t_nand.get_confusion_matrix(X_nand, y_nand)

# Train Perceptron for XOR
t_xor = Perceptron(input_size=2)
t_xor.train(X_xor, y_xor)
xor_accuracy = t_xor.evaluate(X_xor, y_xor)
print("XOR Perceptron Accuracy:", xor_accuracy)
t_xor.plot_loss_curve()
t_xor.get_confusion_matrix(X_xor, y_xor)
