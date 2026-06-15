import pandas as pd 
df = pd.read_csv("titanic.csv")
print(df.isnull().sum())
df["Age"] = df["Age"].fillna(df["Age"].mean())
print(df["Age"].isnull().sum())
df.drop("Cabin", axis=1, inplace=True)
print(df.columns)
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
print(df["Embarked"].isnull().sum())
df["Sex"] = df["Sex"].map({
    "male" : 0,
    "female" : 1
})
print(df["Sex"].head())
df = pd.get_dummies(df, columns=["Embarked"])
print(df.isnull().sum())
df.to_csv("titanic_cleaned.csv", index=False)
