# pca_mnist.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler # Scaled for images
from sklearn.decomposition import PCA

print("=== PCA on MNIST ===")

# 1. Load Data
try:
    mnist = fetch_openml('mnist_784', version=1, as_frame=True, parser='auto')
    X = mnist.data.values.astype(float)
    y = mnist.target.values.astype(int)
except Exception as e:
    print(f"Error loading data: {e}")
    # Fallback if fetch_openml fails in your environment
    print("Data load failed. Ensure you have internet access or a local dataset.")
    exit()

print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

# 2. Preprocessing for Images (MinMax is better than StandardScaler for raw pixels)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply PCA
# We keep 2 components for visualization
pca = PCA(n_components=2)
X_pca_2d = pca.fit_transform(X_scaled)

# 4. Calculate Variance Explained
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print(f"Variance explained by PC1: {explained_var[0]:.4f}")
print(f"Variance explained by PC2: {explained_var[1]:.4f}")
print(f"Total variance captured in 2D: {cumulative_var[1]:.4f}")

# 5. Plotting
plt.figure(figsize=(12, 8))
sns.scatterplot(x=X_pca_2d[:, 0], y=X_pca_2d[:, 1], 
                hue=y, palette="tab10", alpha=0.4, s=15, legend="full")
plt.title(f"PCA on MNIST: 784 Features → 2 Dimensions\n(Total Variance: {cumulative_var[1]:.2%})")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Digit Label", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
