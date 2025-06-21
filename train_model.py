import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load dataset
df = pd.read_csv("minuman_dataset.csv")

# Fitur dan label
X = df[["waktu", "cuaca", "energi", "manis"]]
y = df["minuman"]

# One-hot encoding
X_encoded = pd.get_dummies(X)

# Train model
model = DecisionTreeClassifier()
model.fit(X_encoded, y)

# Simpan model
joblib.dump(model, "decision_tree_model.pkl")

print("✅ Model berhasil disimpan ke decision_tree_model.pkl")
