from fastapi import FastAPI,UploadFile,File
import uvicorn
from PIL import Image
import numpy as np
from io import BytesIO
from tensorflow import keras
import cv2 as cv

app=FastAPI()

model=keras.models.load_model('../1')
# model=keras.models.load_model('../objects/model_final.keras')
classes=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']


def get_image(data) -> np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    return image



@app.post("/predict")
async def predict(file:UploadFile):
    image=get_image(await file.read())
    image=cv.resize(image,(256,256))
    batch_img=np.expand_dims(image,0)
    predictions=model.predict(batch_img)
    # print(predictions)
    maxval=np.max(predictions[0])
    label=classes[np.argmax(predictions[0])]
    # print( {'label':label,'probab':maxval})
    return {'label':label,'probab':str(maxval)}
    

@app.get("/home")
async def home():
    return "hello this is the start of the potato leaf disease project"

if __name__=='__main__':
    uvicorn.run(app,host='localhost',port=8000)