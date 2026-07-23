import pandas as pd

LEAD_TIME_DAYS = 7
Z_95 = 1.645

def run(df: pd.DataFrame, top_n_skus: int = 50) -> pd.DataFrame:
    top_skus = df.groupby("StockCode")["TotalPrice"].sum().nlargest(top_n_skus).index

    daily_by_sku = (
        df[df["StockCode"].isin(top_skus)]
        .groupby(["StockCode", df["InvoiceDate"].dt.date])["Quantity"]
        .sum()
        .reset_index()
    )
    daily_by_sku.columns = ["StockCode", "Date", "Quantity"]

    stats = daily_by_sku.groupby("StockCode")["Quantity"].agg(["mean", "std"]).reset_index()
    stats.columns = ["StockCode", "AvgDailyDemand", "DemandStd"]
    stats["DemandStd"] = stats["DemandStd"].fillna(0)

    stats["Forecast30Day"] = stats["AvgDailyDemand"] * 30
    stats["SafetyStock"] = Z_95 * stats["DemandStd"] * (LEAD_TIME_DAYS ** 0.5)
    stats["ReorderPoint"] = stats["AvgDailyDemand"] * LEAD_TIME_DAYS + stats["SafetyStock"]
    stats["RecommendedOrderQty"] = stats["Forecast30Day"] + stats["SafetyStock"]

    return stats.sort_values("Forecast30Day", ascending=False)