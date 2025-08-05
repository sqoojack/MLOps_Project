from fastapi import FastAPI
import pickle

app = FastAPI()

with open("models/popular_items.pkl", "rb") as f:
    popular_items = pickle.load(f)

@app.get("/recommend")
def recommend(user_id: int):
    return {
        "user_id": user_id,
        "recommended_items": popular_items  # 對所有人都是一樣的
    }

#  uvicorn api.main:app --reload