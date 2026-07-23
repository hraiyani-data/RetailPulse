import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def build_churn_dataset(df: pd.DataFrame, churn_window_days: int = 90) -> pd.DataFrame:
    last_date = df["InvoiceDate"].max()
    cutoff = last_date - pd.Timedelta(days=churn_window_days)

    before = df[df["InvoiceDate"] <= cutoff]
    after = df[df["InvoiceDate"] > cutoff]

    features = before.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (cutoff - x.max()).days),
        Tenure=("InvoiceDate", lambda x: (cutoff - x.min()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
        AvgBasket=("TotalPrice", "mean"),
        DistinctProducts=("StockCode", "nunique"),
        DistinctCountries=("Country", "nunique"),
    ).reset_index()

    churned_ids = set(before["CustomerID"]) - set(after["CustomerID"])
    features["Churned"] = features["CustomerID"].isin(churned_ids).astype(int)

    return features



def train_churn_model(churn_data: pd.DataFrame):
    feature_cols = ["Recency", "Tenure", "Frequency", "Monetary",
                     "AvgBasket", "DistinctProducts", "DistinctCountries"]

    X = churn_data[feature_cols]
    y = churn_data["Churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    return model, X_train, X_test, y_train, y_test, y_proba, auc, feature_cols

def run(df: pd.DataFrame = None) -> pd.DataFrame:
    if df is None:
        df = pd.read_parquet("data/processed/retail_clean.parquet")

    churn_data = build_churn_dataset(df)
    model, X_train, X_test, y_train, y_test, y_proba, auc, feature_cols = train_churn_model(churn_data)

    print("Churn AUC-ROC:", auc)

    joblib.dump(model, "models/xgb_churn_model.pkl")
    joblib.dump(feature_cols, "models/churn_feature_cols.pkl")

    churn_data.to_csv("data/processed/churn_features.csv", index=False)

    return churn_data