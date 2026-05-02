from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_model(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier()

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Model Accuracy: {acc * 100:.2f}%")

    joblib.dump(model, "models/churn_model.pkl")

    return model
