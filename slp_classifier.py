"""
SimpleSLPClassifier - Single Layer Perceptron for Classification
"""

from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from activations import softmax
from slp_base import BaseSLPEstimator, N_ITER_NO_CHANGE, TOL


class SimpleSLPClassifier(BaseSLPEstimator):
    """
    Simple Single Layer Perceptron Classifier with one hidden layer.

    Compatible interface with sklearn.neural_network.MLPClassifier.
    """

    def __init__(
        self,
        hidden_layer_size: int = 100,
        activation: str = "logistic",
        learning_rate: float = 0.01,
        max_iter: int = 200,
        batch_size: int = 32,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize the SLP classifier.

        Parameters:
        -----------
        hidden_layer_size : int
            Number of neurons in the hidden layer
        activation : str
            Activation function ('identity', 'logistic', 'tanh', 'relu'}, default='logistic')
        learning_rate : float
            Learning rate for gradient descent
        max_iter : int
            Maximum number of iterations
        random_state : int or None
            Random seed for reproducibility
        """
        super().__init__(
            hidden_layer_size, activation, learning_rate, max_iter, batch_size, random_state
        )

        # Classifier-specific attributes
        self.classes_: Optional[NDArray[np.int_]] = None  # Unique class labels
        self.n_outputs_: Optional[int] = None  # Number of output neurons

    def _forward_propagation(self, X: NDArray[np.floating]) -> Tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        """
        Perform forward propagation.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data

        Returns:
        --------
        z1, a1, z2, y_pred : tuple of arrays
            Intermediate values for backpropagation
        """
        activation_fn, _ = self._get_activation_function()
        z1 = X @ self.W1_ + self.b1_
        a1 = activation_fn(z1)
        z2 = a1 @ self.W2_ + self.b2_
        y_pred = softmax(z2)
        return z1, a1, z2, y_pred

    def _backward_propagation(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        z1: NDArray[np.floating],
        a1: NDArray[np.floating],
        z2: NDArray[np.floating],
        y_pred: NDArray[np.floating],
    ) -> Tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        """
        Perform backpropagation to compute gradients.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        y : array-like, shape (n_samples, n_outputs)
            One-hot encoded target
        z1, a1, z2, y_pred : arrays
            Values from forward propagation

        Returns:
        --------
        dW1, db1, dW2, db2 : tuple of arrays
            Gradients for weights and biases
        """
        _, activation_derivative_fn = self._get_activation_function()
        n = X.shape[0]
        dz2 = (y_pred - y) / n
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        dz1 = (dz2 @ self.W2_.T) * activation_derivative_fn(z1)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        return dW1, db1, dW2, db2

    def _compute_loss(
        self, y_true: NDArray[np.floating], y_pred: NDArray[np.floating]
    ) -> float:
        """
        Compute cross-entropy loss.

        Parameters:
        -----------
        y_true : array-like, shape (n_samples, n_classes)
            One-hot encoded true labels
        y_pred : array-like, shape (n_samples, n_classes)
            Predicted probabilities

        Returns:
        --------
        loss : float
            Cross-entropy loss
        """
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))

    def fit(
        self, X: NDArray[np.floating], y: NDArray[np.int_]
    ) -> "SimpleSLPClassifier":
        """
        Fit the SLP classifier to training data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target class labels

        Returns:
        --------
        self : object
            Fitted estimator
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.classes_ = np.unique(y)
        self.n_outputs_ = len(self.classes_)
        n_samples = X.shape[0]
        y_onehot = np.zeros((n_samples, self.n_outputs_))
        y_onehot[np.arange(n_samples), np.searchsorted(self.classes_, y)] = 1
        self._initialize_weights(X.shape[1], self.n_outputs_)
        self.loss_curve_ = []
        for _ in range(self.max_iter):
            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                X_batch = X[batch_idx]
                y_batch = y_onehot[batch_idx]
                z1, a1, z2, y_pred = self._forward_propagation(X_batch)
                dW1, db1, dW2, db2 = self._backward_propagation(
                    X_batch, y_batch, z1, a1, z2, y_pred
                )
                dW1, db1, dW2, db2 = self._clip_gradients(dW1, db1, dW2, db2)
                self.W1_ -= self.learning_rate * dW1
                self.b1_ -= self.learning_rate * db1
                self.W2_ -= self.learning_rate * dW2
                self.b2_ -= self.learning_rate * db2
            _, _, _, y_pred_full = self._forward_propagation(X)
            self.loss_curve_.append(self._compute_loss(y_onehot, y_pred_full))
            if len(self.loss_curve_) > N_ITER_NO_CHANGE:
                recent = self.loss_curve_[-N_ITER_NO_CHANGE - 1:]
                if recent[0] - min(recent[1:]) < TOL:
                    break
        return self

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Predict class probabilities for X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples

        Returns:
        --------
        proba : array-like, shape (n_samples, n_classes)
            Class probabilities
        """
        _, _, _, y_pred = self._forward_propagation(X)
        return y_pred

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.int_]:
        """
        Predict class labels for X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples

        Returns:
        --------
        y_pred : array-like, shape (n_samples,)
            Predicted class labels
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: NDArray[np.floating], y: NDArray[np.int_]) -> float:
        """
        Return the mean accuracy on the given test data and labels.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        y : array-like, shape (n_samples,)
            True labels

        Returns:
        --------
        score : float
            Mean accuracy
        """
        return np.mean(self.predict(X) == y)
