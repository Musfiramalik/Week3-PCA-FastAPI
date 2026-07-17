from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import joblib

app = FastAPI()

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the saved model, scaler, and column list
scaler = joblib.load("models/scaler.joblib")
model = joblib.load("models/model.joblib")
model_columns = joblib.load("models/model_columns.joblib")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request, "home.html", {})


@app.get("/dashboard")
def dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html", {})


@app.get("/predict")
def predict_form(request: Request):
    return templates.TemplateResponse(request, "predict.html", {"prediction": None})


@app.post("/predict")
def predict(
    request: Request,
    lagging_reactive: float = Form(...),
    leading_reactive: float = Form(...),
    co2: float = Form(...),
    lagging_pf: float = Form(...),
    leading_pf: float = Form(...),
    nsm: float = Form(...),
    hour: int = Form(...),
    power_factor_ratio: float = Form(...),
    week_status: str = Form(...),
    day_of_week: str = Form(...),
    load_type: str = Form(...),
    month: str = Form(...),
):
    # Build a single-row dataframe with the raw input values
    input_data = pd.DataFrame([{
        "Lagging_Current_Reactive.Power_kVarh": lagging_reactive,
        "Leading_Current_Reactive_Power_kVarh": leading_reactive,
        "CO2(tCO2)": co2,
        "Lagging_Current_Power_Factor": lagging_pf,
        "Leading_Current_Power_Factor": leading_pf,
        "NSM": nsm,
        "Hour": hour,
        "Power_Factor_Ratio": power_factor_ratio,
        "WeekStatus": week_status,
        "Day_of_week": day_of_week,
        "Load_Type": load_type,
        "Month": month,
    }])

    # One-hot encode the same way as training (drop_first=True)
    input_encoded = pd.get_dummies(
        input_data,
        columns=["WeekStatus", "Day_of_week", "Load_Type", "Month"],
        drop_first=True
    )

    # Align columns with training data (add missing columns as 0, in correct order)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Scale the input using the same scaler fit during training
    input_scaled = scaler.transform(input_encoded)

    # Predict
    prediction = model.predict(input_scaled)[0]
    prediction = round(float(prediction), 2)

    return templates.TemplateResponse(
        request, "predict.html", {"prediction": prediction}
    )