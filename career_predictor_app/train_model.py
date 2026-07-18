"""
Run this ONCE (whenever your dataset changes) to train the model and
save it to disk. The Flask app then just loads these files instead of
retraining on every request.

Usage:
    python train_model.py
"""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "student_career_dataset.csv"

# ---------------------------------------------------------------
# 1. Load & clean
# ---------------------------------------------------------------
data = pd.read_csv(DATA_PATH)
data.drop(["Student_ID", "Name"], axis=1, inplace=True)

X = data.drop("Recommended_Career", axis=1)
y = data["Recommended_Career"]

# Save the raw category options BEFORE dummy-encoding, so the web form
# can offer real dropdown choices instead of free text (this fixes the
# "typed value doesn't match a training category" bug).
categorical_cols = X.select_dtypes(include="object").columns.tolist()
category_options = {col: sorted(X[col].dropna().unique().tolist()) for col in categorical_cols}
joblib.dump(category_options, "model/category_options.pkl")

X = pd.get_dummies(X, drop_first=True)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------
# 2. Train & compare models
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results.append((name, acc))
    print(f"{name} Accuracy: {acc:.4f}")

results.sort(key=lambda r: r[1], reverse=True)
print("\nRanked results:")
for name, acc in results:
    print(f"  {name}: {acc:.4f}")

# ---------------------------------------------------------------
# 3. Final model = Random Forest (best performer in your original run)
# ---------------------------------------------------------------
rf = models["Random Forest"]
print("\nRandom Forest classification report:")
print(classification_report(y_test, rf.predict(X_test)))

# ---------------------------------------------------------------
# 4. Persist everything the web app needs
# ---------------------------------------------------------------
joblib.dump(rf, "model/rf_model.pkl")
joblib.dump(le, "model/label_encoder.pkl")
joblib.dump(X.columns.tolist(), "model/model_columns.pkl")

print("\nSaved model/rf_model.pkl, model/label_encoder.pkl, "
      "model/model_columns.pkl, model/category_options.pkl")
