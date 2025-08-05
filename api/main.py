from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load("models/model.pkl")

@app.get("/recommend")
def recommend(user_id: int):
    return {"user_id": user_id, "recommended_items": model.recommend(user_id)}