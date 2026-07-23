import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum")
    ).reset_index()

    return rfm


def scale_rfm(rfm: pd.DataFrame):
    rfm["Frequency_log"] = np.log1p(rfm["Frequency"])
    rfm["Monetary_log"] = np.log1p(rfm["Monetary"])

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency_log", "Monetary_log"]])

    return rfm_scaled, scaler

def choose_k(rfm_scaled, k_range=range(2, 9)) -> dict:
    results = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(rfm_scaled)
        results[k] = {
            "inertia": km.inertia_,
            "silhouette": silhouette_score(rfm_scaled, km.labels_)
        }
    return results

def label_segments(rfm: pd.DataFrame, rfm_scaled, k=5) -> pd.DataFrame:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(rfm_scaled)
    rfm["Cluster"] = km.labels_

    # Har cluster ka centroid (average Recency, Frequency, Monetary) nikalo
    centroids = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()

    # Score banao: Recency kam achha, Frequency/Monetary zyada achha
    centroids["Score"] = (
        centroids["Frequency"].rank() +
        centroids["Monetary"].rank() -
        centroids["Recency"].rank()
    )

    # Score ke hisaab se rank karo, best se worst
    ranked_clusters = centroids["Score"].sort_values(ascending=False).index.tolist()

    segment_names = ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Needs Attention"]
    # Agar k != 5 hai to segment_names ki list utni hi entries honi chahiye jitna k

    label_map = {cluster: segment_names[i] for i, cluster in enumerate(ranked_clusters)}
    rfm["Segment"] = rfm["Cluster"].map(label_map)

    return rfm, km

def run(df: pd.DataFrame = None) -> pd.DataFrame:
    if df is None:
        df = pd.read_parquet("data/processed/retail_clean.parquet")

    rfm = build_rfm(df)
    rfm_scaled, scaler = scale_rfm(rfm)
    rfm, km = label_segments(rfm, rfm_scaled, k=5)

    # Model aur scaler save karo
    joblib.dump(km, "models/kmeans_rfm.pkl")
    joblib.dump(scaler, "models/rfm_scaler.pkl")

    # Final segment table save karo
    rfm.to_csv("data/processed/customer_segments.csv", index=False)

    return rfm

if __name__ == "__main__":
    run()