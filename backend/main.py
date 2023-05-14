from fastapi import FastAPI, File, UploadFile, status
from fastapi.responses import JSONResponse
import pathlib
from PIL import Image
from .model import Model

# Create model
model = Model()

# Create FastAPI app
app = FastAPI()

@app.post("/model/prediction", status_code=200)
async def model_predict(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg")
    if not extension:
        # Return with status code 422 Unprocessable Entity
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": "File must be jpg or jpeg format!"})

    try:
        image = model.preprocess(file)
        result = model.predict(image)
    except Exception as e:
        # Return with status code 500 Internal Server Error
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": str(e)})

    result['filename'] = file.filename

    return JSONResponse(status_code=status.HTTP_200_OK, content={"result": result})
