"""
Base class for SLP Classifier and Regressor
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from activations import (
    relu,
    relu_derivative,
    tanh,
    tanh_derivative,
    logistic,
    logistic_derivative,
)


MAX_GRAD_NORM = 50.0
N_ITER_NO_CHANGE = 10
TOL = 1e-4


class BaseSLPEstimator(ABC):
    """
    Abstract base class for SLP Classifier and Regressor.

    Contains common functionality shared between both estimators.
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
        Initialize the SLP estimator.

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
        self.hidden_layer_size: int = hidden_layer_size
        self.activation: str = activation
        self.learning_rate: float = learning_rate
        self.max_iter: int = max_iter
        self.batch_size: int = batch_size
        self.random_state: Optional[int] = random_state

        # To be initialized in fit()
        self.W1_: Optional[NDArray[np.floating]] = None  # Weights: input -> hidden
        self.b1_: Optional[NDArray[np.floating]] = None  # Biases: hidden layer
        self.W2_: Optional[NDArray[np.floating]] = None  # Weights: hidden -> output
        self.b2_: Optional[NDArray[np.floating]] = None  # Biases: output layer
        self.loss_curve_: list[float] = []  # Track loss during training

        if hidden_layer_size <= 0:
            raise ValueError(f"hidden_layer_size must be > 0, got {hidden_layer_size}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be > 0, got {max_iter}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if activation not in ("relu", "tanh", "logistic"):
            raise ValueError(
                f"activation must be one of 'relu', 'tanh', 'logistic', got '{activation}'"
            )

    def _get_activation_function(
        self,
    ) -> Tuple[Callable[[NDArray], NDArray], Callable[[NDArray], NDArray]]:
        """Return the activation function and its derivative."""
        if self.activation == "relu":
            return relu, relu_derivative
        elif self.activation == "tanh":
            return tanh, tanh_derivative
        elif self.activation == "logistic":
            return logistic, logistic_derivative
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _initialize_weights(self, n_features: int, n_outputs: int) -> None:
        """
        Initialize weights and biases.
        """
        std1 = np.sqrt(2.0 / (n_features + self.hidden_layer_size))
        self.W1_ = np.random.randn(n_features, self.hidden_layer_size) * std1
        self.b1_ = np.zeros(self.hidden_layer_size)
        std2 = np.sqrt(2.0 / (self.hidden_layer_size + n_outputs))
        self.W2_ = np.random.randn(self.hidden_layer_size, n_outputs) * std2
        self.b2_ = np.zeros(n_outputs)

    def _clip_gradients(
        self,
        dW1: NDArray[np.floating],
        db1: NDArray[np.floating],
        dW2: NDArray[np.floating],
        db2: NDArray[np.floating],
    ) -> Tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        """Scale gradients down if their global L2 norm exceeds MAX_GRAD_NORM."""
        grads = [dW1, db1, dW2, db2]
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
        if total_norm > MAX_GRAD_NORM:
            scale = MAX_GRAD_NORM / total_norm
            return tuple(g * scale for g in grads)
        return dW1, db1, dW2, db2

    @abstractmethod
    def _forward_propagation(self, X: NDArray[np.floating]) -> Tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        """
        Perform forward propagation.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
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

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _compute_loss(
        self, y_true: NDArray[np.floating], y_pred: NDArray[np.floating]
    ) -> float:
        """
        Compute loss.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def fit(self, X: NDArray[np.floating], y: NDArray) -> "BaseSLPEstimator":
        """
        Fit the model to training data.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def predict(self, X: NDArray[np.floating]) -> NDArray:
        """
        Make predictions.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def score(self, X: NDArray[np.floating], y: NDArray) -> float:
        """
        Score the model.

        Must be implemented by subclasses.
        """
        pass
