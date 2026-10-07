# Ensemble Methods: Bagging and Boosting

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- BAGGING ----------------
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=10,
    random_state=42
)

bagging.fit(X_train, y_train)
bag_pred = bagging.predict(X_test)

print("Bagging Accuracy:", accuracy_score(y_test, bag_pred))

# ---------------- BOOSTING ----------------
from sklearn.ensemble import GradientBoostingClassifier

boosting = GradientBoostingClassifier(
    n_estimators=50,
    random_state=42
)

boosting.fit(X_train, y_train)
boost_pred = boosting.predict(X_test)

print("Boosting Accuracy:", accuracy_score(y_test, boost_pred))
