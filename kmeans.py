import pandas as pd
from sklearn.cluster import KMeans

data = {
    "Age": [20, 22, 25, 27, 55, 58, 60, 62],
    "Spending": [200, 250, 300, 350, 1000, 1100, 1200, 1300]
}

df = pd.DataFrame(data)

kmeans = KMeans(
    n_clusters=2,
    random_state=42
)

df["Cluster"] = kmeans.fit_predict(df)

print(df)