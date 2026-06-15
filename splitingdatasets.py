from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd 

df = pd.read_csv("titanic_cleaned.csv")


scaler = MinMaxScaler()
df["Age"] = scaler.fit_transform(df[["Age"]])


features = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "SibSp",
    "Parch"
]

X = df[features]


y = df["Survived"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)