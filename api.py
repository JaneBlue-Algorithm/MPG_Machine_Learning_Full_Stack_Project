from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Model ve transformer'ları yükle
model = joblib.load("mpg_poly_model.pkl")
poly = joblib.load("poly_transformer.pkl")
scaler = joblib.load("mpg_scaler.pkl")
feature_columns = joblib.load("mpg_features.pkl")


@app.post("/predict")
def predict(data: dict):

    # Input dataframe oluştur
    input_df = pd.DataFrame(columns=feature_columns)
    input_df.loc[0] = 0

    # Gelen değerleri yerleştir
    input_df.loc[0, "weight"] = data["weight"]
    input_df.loc[0, "acceleration"] = data["acceleration"]
    input_df.loc[0, "model_year"] = data["model_year"]
    input_df.loc[0, "origin_japan"] = data["origin_japan"]
    input_df.loc[0, "origin_usa"] = data["origin_usa"]

    # Polynomial dönüşüm
    input_poly = poly.transform(input_df)

    # Scaling
    input_scaled = scaler.transform(input_poly)

    # Tahmin
    prediction = model.predict(input_scaled)[0]

    return {"predicted_mpg": float(prediction)}
