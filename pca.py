import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = {
    "Age": [20,22,25,27,30],
    "Salary": [20000,25000,30000,35000,40000],
    "Experience": [1,2,3,4,5]
}

df = pd.DataFrame(data)

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

pca = PCA(n_components=2)

principal_components = pca.fit_transform(
    scaled_data
)

pca_df = pd.DataFrame(
    principal_components,
    columns=["PC1", "PC2"]
)

print(pca_df)