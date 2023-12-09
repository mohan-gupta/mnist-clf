import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager

import MySQLdb

from fastapi import FastAPI, UploadFile

import numpy as np
import cv2

from app.predict import get_model, predict

def setup_db():
    # Database configuration
    db_config = {
        'host': os.getenv("MYSQL_HOST"),
        'user': os.getenv("MYSQL_USER"),
        'passwd': os.getenv("MYSQL_PASS"),
        'db': os.getenv("MYSQL_DB"),
    }

    # Create a connection to the database
    conn = MySQLdb.connect(**db_config)
    
    create_table = """
    CREATE TABLE [IF NOT EXISTS] logs(
        image VARCHAR(255) NOT NULL,
        prediction INT NOT NULL
    )"""
    cursor = conn.cursor()
    cursor.execute(create_table)
    conn.commit()
    cursor.close()
    
    return conn

model_dct = {}
mysql_conn = setup_db()

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
    global mysql_conn
    
    # read the image
    contents = await image.read()
    np_img = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # expand to 1 channel
    img = np.expand_dims(img, axis=2)
    
    pred = predict(model_dct['model'], img.copy())
    
    cursor = mysql_conn.cursor()
    query = "INSERT INTO logs (image, prediction) VALUES (%s, %s)"
    
    cursor.execute(query, (contents, pred))
    mysql_conn.commit()
    cursor.close()
    
    return {
        "response": pred
    }