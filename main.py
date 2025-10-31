from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import joblib
import numpy as np
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory="templates")

modelo_diabetes = joblib.load("diabetes_model.pkl")
modelo_seguro = joblib.load("insurance_model.pkl")

@app.get("/", response_class=HTMLResponse)
async def formulario(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "resultado": None})

@app.post("/", response_class=HTMLResponse)
async def predecir(request: Request,
                   modelo: str = Form(...),
                   edad: float = Form(...),
                   bmi: float = Form(...),
                   glucosa: float = Form(0),
                   hijos: int = Form(0),
                   fumador: str = Form("no")):
    if modelo == "diabetes":
        X = np.array([[edad, bmi, glucosa]])
        pred = modelo_diabetes.predict(X)[0]
        prob = modelo_diabetes.predict_proba(X)[0][1]
        res = f"Resultado: {'Positivo' if pred==1 else 'Negativo'} (Probabilidad: {prob*100:.1f}%)"
    else:
        X = pd.DataFrame([[edad, bmi, hijos, 1 if fumador=='sí' else 0]], columns=['age','bmi','children','smoker'])
        costo = modelo_seguro.predict(X)[0]
        res = f"Costo estimado del seguro médico: ${costo:,.0f}"
    return templates.TemplateResponse("result.html", {"request": request, "resultado": res})
