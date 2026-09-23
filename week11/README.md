# Dimensionality Reduction: PCA vs t-SNE on MNIST

This repository demonstrates two fundamental dimensionality reduction techniques—**PCA** and **t-SNE**—applied to the MNIST handwritten digits dataset. It compares how each algorithm handles high-dimensional data (784 pixels per image) and visualizes the results in 2D space.

---

## 📂 Project Structure

The repository contains two separate implementations:

1. **`pca.py/`**: Linear dimensionality reduction using Principal Component Analysis.
2. **`tsne.py/`**: Non-linear dimensionality reduction using t-Distributed Stochastic Neighbor Embedding.

---

## 1️⃣ PCA (Principal Component Analysis)

### 🧠 The Concept
PCA finds new axes (**principal components**) that maximize the variance of the data. It projects the data onto these axes, effectively "squashing" the high-dimensional data into fewer dimensions while keeping as much information as possible.

- **Type:** Linear Transformation
- **Goal:** Maximize Variance / Global Structure
- **Speed:** Very Fast (Deterministic)

### 🖥️ Implementation
The code loads the full MNIST dataset (70,000 images), scales the pixel values, and reduces them from **784 features → 2 components**.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Scale pixels to [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reduce to 2 Dimensions
pca = PCA(n_components=2)
X_pca_2d = pca.fit_transform(X_scaled)

## Results & Visualization
<img width="1192" height="790" alt="download" src="https://github.com/user-attachments/assets/71c1665f-b324-4422-b25c-3ffd6b1c9f68" />
Analysis:

Global Structure: PCA captures the global variance. Digit "1" (orange) forms a distinct crescent because it has significantly less ink than other digits.
Overlap: Digits like 3, 5, and 8 overlap heavily. This happens because PCA only draws straight lines; it cannot untangle curved structures where digits differ by small details (like loops).
Variance: The first two components typically capture ~60% of the total variance.

## 2️⃣ t-SNE (t-distributed Stochastic Neighbor Embedting)
## 🧠 The Concept
t-SNE focuses on local neighborhoods. It asks: "Which points are close to which?" and tries to preserve those relationships in 2D. It does not care about global distances or axes meaning—it cares about keeping neighbors together.

Type: Non-Linear Manifold Learning
Goal: Preserve Local Neighborhoods / Clustering
Speed: Slow (Stochastic/Random)
## 🖥️ Implementation
Since t-SNE is computationally expensive ($O(N^2)$), we sample 5,000 images instead of the full 70,000. We also use init='pca' for faster convergence.

``` python 
from sklearn.manifold import TSNE

# Sample 5,000 points for speed
idx = np.random.choice(X.shape[0], 5000, replace=False)
X_sample = X_scaled[idx]

# Run t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', 
            init='pca', max_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_sample)
```
## 📊 Results & Visualization
<img width="1193" height="989" alt="download" src="https://github.com/user-attachments/assets/ecf7347f-5280-480b-a752-4afd6d04798b" />

Analysis:

Cluster Separation: Unlike PCA, t-SNE creates distinct "islands" for each digit. The clusters are tighter and more separated.
Manifold Unrolling: t-SNE successfully unrolls the non-linear structure of the data. Digits that look similar but have different topology (like 6 and 9) are pushed apart.
Ambiguity Bridges: Notice where clusters touch (e.g., 3 touching 8). These represent images that are genuinely ambiguous even to humans.
