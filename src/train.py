
# train.py
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os

# 讀取特徵
df = pd.read_csv("features/events_processed.csv")

# 簡單熱門推薦: 計算推薦次數
popular_items = df["item_id"].value_counts().head(10).index.tolist()

class PopularityModel:
    def __init__(self, top_items):
        self.top_items = top_items
    def recommend(self, user_id):
        return self.top_items

model = PopularityModel(popular_items)

# MLflow 記錄
mlflow.set_experiment("RetailRocket-Popularity")
with mlflow.start_run():
    mlflow.log_param("model_type", "popularity")
    mlflow.log_metric("top_10_items", len(model.top_items))

    # 存檔
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    mlflow.sklearn.log_model(model, "model")

print("[MLflow] 模型記錄完成")

