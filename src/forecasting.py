import pandas as pd
from prophet import Prophet

def build_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby(df["InvoiceDate"].dt.date)["TotalPrice"].sum().reset_index()
    daily.columns = ["ds", "y"]   # Prophet ko exactly ye column names chahiye
    daily["ds"] = pd.to_datetime(daily["ds"])
    return daily

def mape(y_true, y_pred):
    return (abs((y_true - y_pred) / y_true)).mean() * 100

def backtest(daily: pd.DataFrame, horizon_days: int = 30) -> float:
    train = daily.iloc[:-horizon_days]
    test = daily.iloc[-horizon_days:]

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.add_country_holidays(country_name="UK")
    model.fit(train)

    # Buffer periods rakho taaki missing Saturdays ke baad bhi 30 real din cover ho jaye
    future = model.make_future_dataframe(periods=horizon_days + 15)
    forecast = model.predict(future)

    # Sirf wahi dates match karo jo test me actually maujood hain
    merged = test.merge(forecast[["ds", "yhat"]], on="ds", how="left")

    return mape(merged["y"].values, merged["yhat"].values)

def run(df: pd.DataFrame = None) -> pd.DataFrame:
    if df is None:
        df = pd.read_parquet("data/processed/retail_clean.parquet")

    daily = build_daily_sales(df)

    error = backtest(daily, horizon_days=30)
    print("Backtest MAPE:", error)

    # Ab poore history pe retrain karo, aur asli future 30 din predict karo
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.add_country_holidays(country_name="UK")
    model.fit(daily)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    forecast_result = forecast.tail(30)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    forecast_result.to_csv("data/processed/forecast_30d.csv", index=False)

    fig1 = model.plot(forecast)
    fig1.savefig("reports/figures/forecast_plot.png")
    fig2 = model.plot_components(forecast)
    fig2.savefig("reports/figures/forecast_components.png")

    return forecast_result