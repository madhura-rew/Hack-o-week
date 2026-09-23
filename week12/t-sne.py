# tsne_mnist.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

print("=== t-SNE on MNIST ===")

# 1. Load Data
try:
    mnist = fetch_openml('mnist_784', version=1, as_frame=True, parser='auto')
    X = mnist.data.values.astype(float)
    y = mnist.target.values.astype(int)
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

print(f"Dataset loaded: {X.shape[0]} samples")

# 2. Preprocessing
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 3. Sampling (t-SNE is O(n^2), so we sample 5000 points for speed)
np.random.seed(42)
indices = np.random.choice(X.shape[0], 5000, replace=False)
X_sample = X_scaled[indices]
y_sample = y[indices]
print(f"Running on {len(X_sample)} sampled points...")

# 4. Apply t-SNE
# Using 'auto' learning rate and 'pca' initialization for stability
tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', 
            init='pca', max_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_sample)

# 5. Plotting
plt.figure(figsize=(12, 10))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], 
                hue=y_sample, palette="tab10", alpha=0.7, s=20, legend="full")
plt.title("t-SNE on MNIST: 784 Features → 2 Dimensions\n(Focusing on Local Neighborhoods)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(title="Digit Label", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
