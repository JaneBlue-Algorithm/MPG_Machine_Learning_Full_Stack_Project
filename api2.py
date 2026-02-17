# from fastapi import FastAPI
# import joblib
# import pandas as pd

# app = FastAPI()

# pipeline = joblib.load("mpg_pipeline.pkl")

# @app.post("/predict")
# def predict(data: dict):

#     try:
#         input_df = pd.DataFrame([data])
#         prediction = pipeline.predict(input_df)[0]
#         return {"predicted_mpg": float(prediction)}

#     except Exception as e:
#         return {"error": str(e)}


# Bu kodu terminalde calistir -> 
# python -m uvicorn api:app --reload

# Bu da tarayici
# http://127.0.0.1:8000/docs



from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

pipeline = joblib.load("mpg_pipeline.pkl")


# Input schema
class CarFeatures(BaseModel):
    cylinders: int
    displacement: float
    horsepower: float
    weight: float
    acceleration: float
    model_year: int
    origin: str


@app.post("/predict")
def predict(data: CarFeatures):

    input_df = pd.DataFrame([data.dict()])

    prediction = pipeline.predict(input_df)[0]

    return {"predicted_mpg": float(prediction)}
