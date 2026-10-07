import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# Create data
np.random.seed(42)

X = np.linspace(0, 10, 20)
y = np.sin(X) + np.random.normal(0, 0.2, 20)

X = X.reshape(-1, 1)

# ---------------- UNDERFITTING ----------------
underfit = make_pipeline(
    PolynomialFeatures(degree=1),
    LinearRegression()
)

underfit.fit(X, y)
y_under = underfit.predict(X)

# ---------------- OVERFITTING ----------------
overfit = make_pipeline(
    PolynomialFeatures(degree=15),
    LinearRegression()
)

overfit.fit(X, y)
y_over = overfit.predict(X)

# ---------------- REGULARIZATION ----------------
regularized = make_pipeline(
    PolynomialFeatures(degree=15),
    Ridge(alpha=10)
)

regularized.fit(X, y)
y_regularized = regularized.predict(X)

# Print errors
print("Underfitting MSE:",
      mean_squared_error(y, y_under))

print("Overfitting MSE:",
      mean_squared_error(y, y_over))

print("Regularized MSE:",
      mean_squared_error(y, y_regularized))

# Plot
plt.scatter(X, y, label="Data")

plt.plot(X, y_under, label="Underfitting")
plt.plot(X, y_over, label="Overfitting")
plt.plot(X, y_regularized, label="Regularized")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Underfitting vs Overfitting vs Regularization")
plt.legend()
plt.show()
