# PCA on MNIST Handwritten Digits

## 📋 Overview
This project applies **Principal Component Analysis (PCA)** to the MNIST dataset, reducing handwritten digit images from **784 pixel features down to 2 dimensions** for visualization.

PCA is a **linear**, deterministic dimensionality reduction technique that projects data onto new orthogonal axes (principal components) chosen to maximize variance.

## 🧠 How It Works
1. **Load** the MNIST dataset (70,000 images × 784 pixels).
2. **Scale** pixel values to `[0, 1]` using `MinMaxScaler`.
3. **Compute** the top 2 principal components via eigendecomposition of the covariance matrix.
4. **Project** all data onto these 2 axes.
5. **Measure** how much variance is retained.

## 💻 Implementation

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Scale pixels to [0, 1] — preferred over StandardScaler for image data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reduce 784 features → 2 dimensions
pca = PCA(n_components=2)
X_pca_2d = pca.fit_transform(X_scaled)

# Variance captured by the retained components
print(pca.explained_variance_ratio_.sum())
```
## 📊 Results
<img width="989" height="790" alt="download" src="https://github.com/user-attachments/assets/9dbe3db9-9fe4-4272-b115-70dd0f3d33a1" />
Observations: 
Digit "1" separates cleanly (orange crescent on the left). It has far less ink than other digits, so it dominates PC1.
Digit "0" spreads widest along PC1, reflecting high variation in size and roundness.
Digits 2–9 overlap heavily in the center. PCA only draws straight lines, so it cannot untangle curved structure where digits differ by small details like loops.
The first two components capture roughly 60% of total variance — a lot of information is lost when compressing to 2D.
