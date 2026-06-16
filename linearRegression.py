import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data = {
    "hours":[1,2,3,4,5,6],
    "marks":[20,30,40,50,60,70]
}
df = pd.DataFrame(data)
x = df[["hours"]]
y= df[["marks"]]
X_train, X_test, y_train, y_test =train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)
model = LinearRegression()
model.fit(X_train,y_train)
result = model.predict(pd.DataFrame({"hours":[7]}))
print("predicted marks:" ,result)