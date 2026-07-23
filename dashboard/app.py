import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="RetailPulse", layout="wide")

@st.cache_data
def load_segments():
    return pd.read_csv("data/processed/customer_segments.csv")

@st.cache_data
def load_churn():
    return pd.read_csv("data/processed/churn_features.csv")

@st.cache_data
def load_inventory():
    return pd.read_csv("data/processed/inventory_recommendation.csv")

@st.cache_data
def load_forecast():
    return pd.read_csv("data/processed/forecast_30d.csv", parse_dates=["ds"])

# --- Data Load (cached, taaki har click par reload na ho) ---
@st.cache_data
def load_data():
    df = pd.read_parquet("data/processed/retail_clean.parquet")
    return df

df = load_data()
segments = load_segments()
st.sidebar.title("📊 RetailPulse")
st.sidebar.caption("AI-Powered Customer Analytics & Demand Forecasting")

page = st.sidebar.radio("Navigate", [
    "Executive Summary",
    "Sales Analytics",
    "Customer Segmentation",
    "Demand Forecasting",
    "Churn Prediction",
    "Inventory Optimization",
    "Project Notes",
])

# --- Global filters ---
st.sidebar.markdown("---")
st.sidebar.subheader("Global Filters")
countries = st.sidebar.multiselect("Country", options=sorted(df["Country"].unique()))
min_d, max_d = df["InvoiceDate"].min().date(), df["InvoiceDate"].max().date()
date_range = st.sidebar.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)

filtered = df.copy()
if countries:
    filtered = filtered[filtered["Country"].isin(countries)]
if len(date_range) == 2:
    filtered = filtered[
        (filtered["InvoiceDate"].dt.date >= date_range[0]) &
        (filtered["InvoiceDate"].dt.date <= date_range[1])
    ]

# --- Page 1: Executive Summary ---
if page == "Executive Summary":
    st.title("📊 RetailPulse — Executive Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"£{filtered['TotalPrice'].sum():,.0f}")
    col2.metric("Total Orders", f"{filtered['InvoiceNo'].nunique():,}")
    col3.metric("Unique Customers", f"{filtered['CustomerID'].nunique():,}")

    st.subheader("Daily Revenue Trend")
    daily = filtered.groupby(filtered["InvoiceDate"].dt.date)["TotalPrice"].sum()
    st.line_chart(daily)

    st.subheader("Top 10 Countries by Revenue")
    top_countries = filtered.groupby("Country")["TotalPrice"].sum().nlargest(10)
    st.bar_chart(top_countries)

# --- Page 2: Sales Analytics ---
elif page == "Sales Analytics":
    st.title("📈 Sales Analytics")

    st.subheader("Top 10 Products by Revenue")
    top_products = filtered.groupby("Description")["TotalPrice"].sum().nlargest(10)
    st.bar_chart(top_products)

    st.subheader("Revenue by Weekday")
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    weekday_rev = filtered.groupby("Weekday")["TotalPrice"].sum().reindex(order)
    st.bar_chart(weekday_rev)

    st.subheader("Revenue by Hour of Day")
    hourly_rev = filtered.groupby("Hour")["TotalPrice"].sum()
    st.bar_chart(hourly_rev)

# --- Page 3: Customer Segmentation ---
elif page == "Customer Segmentation":
    st.title("👥 Customer Segmentation (RFM + KMeans)")

    segments = load_segments()

    st.subheader("Segment Sizes")
    seg_counts = segments["Segment"].value_counts()
    st.bar_chart(seg_counts)

    st.subheader("Segment Profile (Average RFM)")
    profile = segments.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean()
    st.dataframe(profile.style.format("{:.1f}"))

    st.subheader("RFM Scatter — Recency vs Monetary")
    fig, ax = plt.subplots(figsize=(8, 5))
    for seg in segments["Segment"].unique():
        subset = segments[segments["Segment"] == seg]
        ax.scatter(subset["Recency"], subset["Monetary"], label=seg, alpha=0.5, s=10)
    ax.set_xlabel("Recency (days)")
    ax.set_ylabel("Monetary (£)")
    ax.legend()
    st.pyplot(fig)

# --- Page 4: Demand Forecasting ---
elif page == "Demand Forecasting":
    st.title("🔮 Demand Forecasting (Prophet)")

    forecast = load_forecast()

    st.subheader("Next 30-Day Revenue Forecast")
    st.line_chart(forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]])

    st.subheader("Full History — Actual vs Model Fit (Backtest View)")
    st.image("reports/figures/forecast_plot.png", caption="Prophet fit across full history with confidence band")

    st.subheader("Forecast Table")
    st.dataframe(forecast)

    st.caption("Backtest MAPE: 26.85% (last 30 real days, holdout-validated)")

# --- Page 5: Churn Prediction ---
elif page == "Churn Prediction":
    st.title("⚠️ Customer Churn Prediction (XGBoost)")

    churn = load_churn()

    col1, col2 = st.columns(2)
    col1.metric("Overall Churn Rate", f"{churn['Churned'].mean()*100:.1f}%")
    col2.metric("Total Customers Analyzed", f"{len(churn):,}")

    st.subheader("Model Performance")
    st.write("AUC-ROC: **0.781** &nbsp;&nbsp; Precision@Top20% Risk: **0.82**")

    st.image("reports/figures/churn_feature_importance.png", caption="Feature Importance")
    st.image("reports/figures/churn_confusion_matrix.png", caption="Confusion Matrix")

    st.subheader("Top 20 At-Risk Customers")
    at_risk = churn[churn["Churned"] == 1].sort_values("Monetary", ascending=False).head(20)
    st.dataframe(at_risk[["CustomerID", "Recency", "Frequency", "Monetary", "Churned"]])

# --- Page 6: Inventory Optimization ---
elif page == "Inventory Optimization":
    st.title("📦 Inventory Optimization")

    inventory = load_inventory()

    st.subheader("Reorder Recommendations (Top 50 SKUs)")
    st.dataframe(inventory)

    st.subheader("Recommended Order Quantity — Top 15 SKUs")
    top15 = inventory.head(15).set_index("StockCode")["RecommendedOrderQty"]
    st.bar_chart(top15)

# --- Page 7: Project Notes ---
elif page == "Project Notes":
    st.title("📑 Project Notes")

    st.markdown("""
    ### Honest Results Summary

    | Model | Metric | Result | Target |
    |---|---|---|---|
    | Segmentation | Segments found | 5 | 6-8 |
    | Forecasting | Backtest MAPE | 26.85% | ≤12% |
    | Churn | AUC-ROC | 0.781 | ≥0.88 |
    | Churn | Precision@Top20% | 0.82 | ≥0.75 ✅ |

    ### Challenges Faced
    - Prophet's `make_future_dataframe()` generates continuous calendar dates,
      but this retailer has no Saturday transactions — caused a row-position
      mismatch in backtesting until fixed with a date-based merge.
    - Churn model required strict cutoff-date splitting to avoid data leakage
      between features and labels.

    ### Future Roadmap
    - Category-level forecasting to reduce MAPE
    - Optuna hyperparameter tuning for churn model
    - Docker/Airflow/MLflow production deployment
    """)