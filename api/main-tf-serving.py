import uvicorn
from fastapi import FastAPI, File, UploadFile
import numpy
import tensorflow as tf
from PIL import Image
from io import BytesIO
import requests


app = FastAPI()

endpoint="http://localhost:8502/v1/models/potato_disease_model:predict"

CLASS_NAMES=["Early Blight","Late Blight","Healthy"]

def load_image_into_numpy_array(data)->numpy.ndarray:
    image = numpy.array(Image.open(BytesIO(data)))
    return image

@app.get("/ping")
async def ping():
    return "Hi, Hello I am alive"

@app.post("/predict")
async def predict(file: UploadFile=File(...)): # Use await here to read the file asynchronously
    image_data = load_image_into_numpy_array(await file.read())
    img_batch = numpy.expand_dims(image_data, 0)
    json_data={
        "instances":img_batch.tolist()
    }
    response=requests.post(endpoint,json=json_data)
    prediction=numpy.array(response.json()["predictions"][0])
    predicted_class=CLASS_NAMES[numpy.argmax(prediction)]
    confidence=numpy.max(prediction)

    return {
        "class":predicted_class,
        "confidence":confidence
    }




if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
