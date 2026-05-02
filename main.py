from src.data_preprocessing import preprocess_data
from src.model_training import train_model
from src.prediction import load_model, predict

X_train, X_test, y_train, y_test = preprocess_data("data/churn_data.csv")

model = train_model(X_train, y_train, X_test, y_test)

model = load_model()

predictions = predict(model, X_test)

print("Predictions:", predictions[:10])
