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
#  之後在瀏覽器後面加: /recommend?user_id=257957
#  ex: http://127.0.0.1:8000/recommend?user_id=257957