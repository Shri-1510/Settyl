from fastapi import FastAPI, Query
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle


class OutputData(BaseModel):
    internal_status: str

with open('svm_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

app = FastAPI()

@app.get("/predict", response_model=OutputData)
async def predict(external_status: str = Query(...)):
    external_status_encoded = label_encoder.transform([external_status])

    internal_status_encoded = model.predict(external_status_encoded.reshape(-1, 1))

    internal_status = label_encoder.inverse_transform(internal_status_encoded)[0]

    return {"internal_status": internal_status}


