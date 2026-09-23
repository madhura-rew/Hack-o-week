
# t-SNE on MNIST Handwritten Digits

## 📋 Overview
This project applies **t-Distributed Stochastic Neighbor Embedding (t-SNE)** to the MNIST dataset, producing a 2D map where visually similar digits cluster together.

t-SNE is a **non-linear**, stochastic manifold-learning technique designed specifically for **visualization**. Rather than preserving global distances, it preserves local neighborhoods — keeping neighbors close and pushing unrelated points apart.

## 🧠 How It Works
1. **Load** MNIST and **sample 5,000 images** (t-SNE is $O(N^2)$).
2. **Scale** pixel values to `[0, 1]`.
3. **Compute pairwise affinities** in high-D using a Gaussian kernel, with bandwidth set by `perplexity`.
4. **Define low-D affinities** using a Student-t distribution (heavy tails prevent the crowding problem).
5. **Minimize KL divergence** between the two distributions via gradient descent.

## 💻 Implementation

```python
from sklearn.manifold import TSNE

# Sample 5,000 points for tractable runtime
idx = np.random.choice(X.shape[0], 5000, replace=False)
X_sample = X_scaled[idx]

# init='pca' gives stable convergence; max_iter replaces old n_iter
tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto',
            init='pca', max_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_sample)
```
## 📊 Results
<img width="1193" height="989" alt="download" src="https://github.com/user-attachments/assets/87f7eeaf-e46c-45b5-8998-50f6431d7cc0" />

Observations
Distinct islands emerge. Unlike PCA's single blob, each digit forms its own tight cluster.
Non-linear unrolling works. Digits differing only by topology (e.g. 6 vs 9) get pushed into separate regions.
Bridges between clusters are meaningful. Where 3 touches 8, or 4 touches 9, those images are genuinely ambiguous even to human eyes.
Noise outliers scattered between clusters are usually poorly written or mislabeled samples — useful for data cleaning.
