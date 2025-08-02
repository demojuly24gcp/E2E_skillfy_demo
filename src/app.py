from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()
model = pickle.load(open("model/model.pkl", "rb"))

class Input(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(input: Input):
    data = [[
        input.sepal_length, input.sepal_width,
        input.petal_length, input.petal_width
    ]]
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}
