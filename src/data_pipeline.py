import pandas as pd

def load_raw() -> pd.DataFrame:
    dtypes = {
        "Invoice": str,
        "StockCode": str,
        "Description": str,
        "Customer ID": "Int64",   # Int64 (capital I) = nullable integer, NaN allow karta hai
        "Country": str,
    }
    df1 = pd.read_csv("data/raw/online_retail_2009-2010.csv", encoding="utf-8-sig", dtype=dtypes)
    df2 = pd.read_csv("data/raw/online_retail_2010-2011.csv", encoding="utf-8-sig", dtype=dtypes)
    return pd.concat([df1, df2], ignore_index=True)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
        "Invoice": "InvoiceNo",
        "Price": "UnitPrice",
        "Customer ID": "CustomerID"
    })

def clean(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Cancelled orders hatao (Quantity filter se PEHLE)
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # 2. Missing CustomerID wali rows hatao
    df = df.dropna(subset=["CustomerID"])

    # 3. Sirf positive Quantity aur Price rakho
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # 4. Duplicate rows hatao
    df = df.drop_duplicates()

    # 5. InvoiceDate ko proper datetime banao
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # 6. Extreme outliers cap karo (99.9th percentile)
    q_cap = df["Quantity"].quantile(0.999)
    df["Quantity"] = df["Quantity"].clip(upper=q_cap)

    # 7. Naye useful columns banao
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["Weekday"] = df["InvoiceDate"].dt.day_name()
    df["Hour"] = df["InvoiceDate"].dt.hour

    return df

def run():
    df = load_raw()
    df = standardize_columns(df)
    df = clean(df)
    print("Before saving, shape:", df.shape)

    df.to_parquet("data/processed/retail_clean.parquet", index=False)
    print("Parquet saved.")

    df.to_csv("data/processed/retail_clean.csv", index=False)
    print("CSV saved.")

    return df

if __name__ == "__main__":
    run()