import cv2
import numpy as np
from fastapi import FastAPI, File

from face_recognizer import FaceRecognizer
face_rec = FaceRecognizer()
face_rec.create_known_faces('data/mask_nomask')

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.post("/update")
def update():
    face_rec.create_known_faces('data/mask_nomask')
    return {"result": 'ok'}

@app.post("/file")
def file(file: bytes = File(...)):
    image_array = np.frombuffer(file, dtype=np.uint8)  # numpy array
    img = cv2.imdecode(image_array, cv2.COLOR_RGBA2BGR)
    result = face_rec.recognize(img)
    return {"result": result}
