from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
import os

# Dummy training data (2 features)
X = np.random.rand(100, 2)
y = np.random.randint(0, 2, 100)

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Create models folder
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, "models/churn_model.pkl")

print("✅ churn_model.pkl created")
