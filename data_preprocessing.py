import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(path):
    df = pd.read_csv(path)

    df = df.dropna()

    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    target = 'Churn'

    X = df.drop(target, axis=1)
    y = df[target]

    return train_test_split(X, y, test_size=0.2, random_state=42)
