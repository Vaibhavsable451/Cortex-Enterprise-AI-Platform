"""End-to-end tabular ML pipeline with MLflow tracking.
Usage: python -m ml.pipeline [path.csv target_column]   (defaults to sklearn breast-cancer data)
"""
import sys

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load(path=None, target="target"):
    if path:
        return pd.read_csv(path), target
    from sklearn.datasets import load_breast_cancer

    d = load_breast_cancer(as_frame=True)
    return d.frame, "target"


def validate(df: pd.DataFrame, target: str):
    assert target in df.columns, "target missing"
    assert len(df) > 50, "too few rows"
    return df.drop_duplicates().dropna(subset=[target])


def engineer(df: pd.DataFrame, target: str):
    X, y = df.drop(columns=[target]), df[target]
    X = X.select_dtypes(include=np.number)
    X = X.fillna(X.median())
    return X, y


def main(path=None, target="target", k=15):
    df, target = load(path, target)
    df = validate(df, target)
    X, y = engineer(df, target)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipe = Pipeline([("scale", StandardScaler()), ("select", SelectKBest(f_classif, k=min(k, X.shape[1]))),
                     ("clf", RandomForestClassifier(n_estimators=300, random_state=42))])
    mlflow.set_experiment("tabular-baseline")
    with mlflow.start_run():
        mlflow.log_params({"k_features": k, "n_estimators": 300})
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        mlflow.log_metrics({"accuracy": accuracy_score(yte, pred), "f1": f1_score(yte, pred, average="macro")})
        mlflow.sklearn.log_model(pipe, "model")
        print(classification_report(yte, pred))


if __name__ == "__main__":
    main(*(sys.argv[1:3]))
