import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

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

@st.cache_resource
def load_churn_model():
    model = joblib.load("models/xgb_churn_model.pkl")
    feature_cols = joblib.load("models/churn_feature_cols.pkl")
    return model, feature_cols

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
    daily = filtered.groupby(filtered["InvoiceDate"].dt.date)["TotalPrice"].sum().reset_index()
    daily.columns = ["Date", "Revenue"]
    fig = px.line(daily, x="Date", y="Revenue", title=None)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 Countries by Revenue")
    top_countries = filtered.groupby("Country")["TotalPrice"].sum().nlargest(10).reset_index()
    fig = px.bar(top_countries, x="TotalPrice", y="Country", orientation="h",
                 labels={"TotalPrice": "Revenue (£)"})
    fig.update_layout(yaxis=dict(categoryorder="total ascending"), margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

# --- Page 2: Sales Analytics ---
elif page == "Sales Analytics":
    st.title("📈 Sales Analytics")

    st.subheader("Top 10 Products by Revenue")
    top_products = filtered.groupby("Description")["TotalPrice"].sum().nlargest(10).reset_index()
    fig = px.bar(top_products, x="TotalPrice", y="Description", orientation="h",
                 labels={"TotalPrice": "Revenue (£)", "Description": "Product"})
    fig.update_layout(yaxis=dict(categoryorder="total ascending"), margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Revenue by Weekday")
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    weekday_rev = filtered.groupby("Weekday")["TotalPrice"].sum().reindex(order).reset_index()
    weekday_rev.columns = ["Weekday", "Revenue"]
    fig = px.bar(weekday_rev, x="Weekday", y="Revenue")
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Revenue by Hour of Day")
    hourly_rev = filtered.groupby("Hour")["TotalPrice"].sum().reset_index()
    hourly_rev.columns = ["Hour", "Revenue"]
    fig = px.bar(hourly_rev, x="Hour", y="Revenue")
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

# --- Page 3: Customer Segmentation ---
elif page == "Customer Segmentation":
    st.title("👥 Customer Segmentation (RFM + KMeans)")

    segments = load_segments()

    st.subheader("Segment Sizes")
    seg_counts = segments["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]
    fig = px.bar(seg_counts, x="Segment", y="Count", color="Segment")
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment Profile (Average RFM)")
    profile = segments.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean()
    st.dataframe(profile.style.format("{:.1f}"))

    st.subheader("RFM Scatter — Recency vs Monetary")
    st.caption("Hover over any point to see that customer's ID, frequency, and segment.")
    fig = px.scatter(
        segments, x="Recency", y="Monetary", color="Segment",
        hover_data=["CustomerID", "Frequency"],
        labels={"Recency": "Recency (days)", "Monetary": "Monetary (£)"},
        opacity=0.6,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

# --- Page 4: Demand Forecasting ---
elif page == "Demand Forecasting":
    st.title("🔮 Demand Forecasting (Prophet)")

    forecast = load_forecast()

    st.subheader("Next 30-Day Revenue Forecast")
    st.caption("Hover over the chart to see exact forecast values and confidence bounds for any date.")
    fig = px.line(forecast, x="ds", y=["yhat", "yhat_lower", "yhat_upper"],
                  labels={"ds": "Date", "value": "Revenue (£)", "variable": "Series"})
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Full History — Actual vs Model Fit (Backtest View)")
    st.image("reports/figures/forecast_plot.png", caption="Prophet fit across full history with confidence band")

    st.subheader("Forecast Table")
    st.dataframe(forecast)

    st.caption("Backtest MAPE: 26.85% (last 30 real days, holdout-validated)")

# --- Page 5: Churn Prediction ---
elif page == "Churn Prediction":
    st.title("⚠️ Customer Churn Prediction (XGBoost)")

    churn = load_churn()
    model, feature_cols = load_churn_model()

    col1, col2 = st.columns(2)
    col1.metric("Overall Churn Rate", f"{churn['Churned'].mean()*100:.1f}%")
    col2.metric("Total Customers Analyzed", f"{len(churn):,}")

    st.subheader("Model Performance")
    st.write("AUC-ROC: **0.781** &nbsp;&nbsp; Precision@Top20% Risk: **0.82**")

    st.subheader("Feature Importance")
    st.caption("Hover over any bar to see the exact importance score.")
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)
    fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h")
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

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
    st.caption("Hover over any bar to see the exact recommended order quantity.")

    top15 = inventory.head(15).copy()
    top15["StockCode"] = top15["StockCode"].astype(str)   # SKU ko text treat karo, number nahi
    top15 = top15.sort_values("RecommendedOrderQty", ascending=True)

    fig = px.bar(
        top15,
        x="RecommendedOrderQty",
        y="StockCode",
        orientation="h",
        text="RecommendedOrderQty",
        labels={"RecommendedOrderQty": "Recommended Order Qty", "StockCode": "SKU"},
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_layout(
        yaxis=dict(type="category", categoryorder="array", categoryarray=top15["StockCode"].tolist()),
        margin=dict(l=0, r=40, t=10, b=0),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

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