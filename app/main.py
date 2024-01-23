from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile

import numpy as np
import cv2

from app.predict import get_model, predict

model_dct = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    model_dct['model'] = get_model()
    yield
    # Clean up the ML models and release the resources
    model_dct.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def home():
    return {"data": "Server is running!"}

@app.post("/predict")
async def pred(image: UploadFile):
    # read the image
    contents = await image.read()
    np_img = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # expand to 1 channel
    img = np.expand_dims(img, axis=2)
    
    pred = predict(model_dct['model'], img.copy())
    return {
        "response": pred
    }