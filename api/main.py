import uvicorn
from fastapi import FastAPI, File, UploadFile
import numpy
import tensorflow as tf
from PIL import Image
from io import BytesIO

from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
MODEL=tf.keras.models.load_model("../saved_models/1")
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
    predictions=MODEL.predict(img_batch)
    predicted_class=CLASS_NAMES[numpy.argmax(predictions[0])]
    confidence=numpy.max(predictions[0])
    return {
        'class':predicted_class,
        'confidence':float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
