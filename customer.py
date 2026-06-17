import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

data = {
    "Age":[20,22,25,27,30,50,55,58,60,62],
    "Income":[20000,25000,30000,35000,40000,
              80000,90000,95000,100000,110000],
    "Spending":[200,250,300,350,400,
                1000,1100,1200,1300,1400]
}

df = pd.DataFrame(data)

X = df[["Income","Spending"]]

kmeans = KMeans(
    n_clusters=2,
    random_state=42
)

df["Cluster"] = kmeans.fit_predict(X)

print(df)

plt.scatter(
    df["Income"],
    df["Spending"],
    c=df["Cluster"]
)

plt.xlabel("Income")
plt.ylabel("Spending")
plt.title("Customer Segmentation")

plt.show()