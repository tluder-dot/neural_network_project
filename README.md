# Single-Layer Perceptron (SLP)

From-scratch NumPy implementation of a one-hidden-layer neural network for binary classification, multi-class classification, and regression. Follows the scikit-learn estimator interface (`fit`, `predict`, `score`).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the test suite:

```bash
pytest tests/
```

Explore the demonstration notebook:

```bash
jupyter notebook report.ipynb
```

The notebook compares this implementation against scikit-learn's `MLPClassifier` and `MLPRegressor` on four datasets (diabetes, breast cancer, iris, California housing).

## Project Structure

```
activations.py        Activation functions (ReLU, tanh, logistic, softmax)
slp_base.py           Abstract base estimator with shared training logic
slp_classifier.py     SimpleSLPClassifier (softmax + cross-entropy)
slp_regressor.py      SimpleSLPRegressor (linear output + MSE)
tests/                Test suite (54 tests)
report.ipynb          Demonstration notebook
```
