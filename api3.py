# front end icin 
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi import Form

from fastapi.staticfiles import StaticFiles


# back end icin
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

pipeline = joblib.load("mpg_pipeline.pkl")
feature_columns = joblib.load("features.pkl")


class CarFeatures(BaseModel):
    weight: float
    acceleration: float
    model_year: int
    origin_japan: int
    origin_usa: int


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.post("/predict_form", response_class=HTMLResponse)
def predict_form(
    request: Request,
    weight: float = Form(...),
    acceleration: float = Form(...),
    model_year: int = Form(...),
    origin_japan: int = Form(...),
    origin_usa: int = Form(...)
):
    df = pd.DataFrame([{
        "weight": weight,
        "acceleration": acceleration,
        "model_year": model_year,
        "origin_japan": origin_japan,
        "origin_usa": origin_usa
    }])

    df = df[feature_columns]

    prediction = pipeline.predict(df)[0]
    prediction = round(prediction, 2)

    # 🔥 Yorumlama kısmı
    if prediction < 20:
        comment = "Düşük yakıt verimliliği 🚨"
        alert_class = "alert-danger"
    elif prediction < 30:
        comment = "Orta seviye yakıt verimliliği ⚖️"
        alert_class = "alert-warning"
    else:
        comment = "Yüksek yakıt verimliliği ✅"
        alert_class = "alert-success"

    return templates.TemplateResponse(
        "index.html",
        {
        "request": request,
        "prediction": prediction,
        "comment": comment,
        "alert_class": alert_class,
        "weight": weight,
        "acceleration": acceleration,
        "model_year": model_year,
        "origin_japan": origin_japan,
        "origin_usa": origin_usa
    }
    )
# ------------------------------------------------------------------

# @app.post("/predict")
# def predict(data: CarFeatures):

#     df = pd.DataFrame([data.dict()])
#     df = df[feature_columns]  # kolon sırası garanti

#     prediction = pipeline.predict(df)[0]

#     return {"predicted_mpg": float(prediction)}


# python -m uvicorn api3:app --reload
# swagger da kullanacagim data 
# {
#   "weight": 2000.0,
#   "acceleration": 16.0,
#   "model_year": 82,
#   "origin_japan": 1,
#   "origin_usa": 0
# }