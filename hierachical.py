import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

data = {
    "Age": [20, 22, 25, 55, 58, 60],
    "Spending": [200, 250, 300, 1000, 1100, 1200]
}

df = pd.DataFrame(data)

# Create hierarchy
Z = linkage(df, method="ward")

# Draw dendrogram
plt.figure(figsize=(8,5))

dendrogram(Z)

plt.title("Hierarchical Clustering")
plt.show()

# Create clusters
model = AgglomerativeClustering(
    n_clusters=2
)

df["Cluster"] = model.fit_predict(df)

print(df)