# Install first if needed:
# pip install xgboost

from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

model = XGBClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1
)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("XGBoost Accuracy:", accuracy_score(y_test, prediction))
