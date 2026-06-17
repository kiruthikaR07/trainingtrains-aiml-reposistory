import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = {
    "Hours": [1, 2, 3, 4, 5, 6],
    "Pass": [0, 0, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[["Hours"]]
y = df["Pass"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = LogisticRegression()


model.fit(X_train, y_train)


result = model.predict(pd.DataFrame({"Hours":[5]}))

print("Prediction:", result[0])