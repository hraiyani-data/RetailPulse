import pandas as pd
from src import data_pipeline, segmentation, forecasting, churn_prediction, inventory_optimization

def main():
    print("Step 1: Data pipeline...")
    df = data_pipeline.run()

    print("Step 2: Segmentation...")
    segmentation.run(df)

    print("Step 3: Forecasting...")
    forecasting.run(df)

    print("Step 4: Churn prediction...")
    churn_prediction.run(df)

    print("Step 5: Inventory optimization...")
    inventory = inventory_optimization.run(df)
    inventory.to_csv("data/processed/inventory_recommendation.csv", index=False)

    print("Pipeline complete.")

if __name__ == "__main__":
    main()