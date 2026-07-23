import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.churn_prediction import build_churn_dataset, train_churn_model

df = pd.read_parquet("data/processed/retail_clean.parquet")
churn_data = build_churn_dataset(df)

model, X_train, X_test, y_train, y_test, y_proba, auc, feature_cols = train_churn_model(churn_data)

y_pred = (y_proba >= 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Active", "Churned"])
disp.plot()
plt.title("Churn Model — Confusion Matrix")
plt.savefig("reports/figures/churn_confusion_matrix.png")
plt.close()

print("Confusion matrix saved.")