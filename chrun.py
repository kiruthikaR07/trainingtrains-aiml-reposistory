import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = {
    "Age":[20,25,30,35,40,45,50,55],
    "MonthlyCharges":[500,700,900,1200,1500,1700,2000,2200],
    "Tenure":[2,5,8,15,20,25,30,35],
    "Churn":[1,1,1,0,0,0,0,0]
}

df = pd.DataFrame(data)

X = df[[
    "Age",
    "MonthlyCharges",
    "Tenure"
]]

y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

new_customer = pd.DataFrame({
    "Age":[23],
    "MonthlyCharges":[600],
    "Tenure":[3]
})

result = model.predict(
    new_customer
)

print("Prediction:", result[0])

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("Accuracy:", accuracy)