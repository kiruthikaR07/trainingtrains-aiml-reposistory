import pandas as pd 
import matplotlib.pyplot as plt
df = pd.read_csv("titanic.csv")
print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.dtypes)
print(df["Survived"].value_counts())
df["Survived"].value_counts().plot(kind="bar")
plt.title("Survival Count")
plt.show()