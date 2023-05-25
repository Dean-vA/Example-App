from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from model import load_model 
from load_data import load_and_preprocess_image
from predict import predict_digit

class ImagePath(BaseModel):
    """
    Path to an image file.

    Attributes:
        path_to_img (str): Path to the image file.
    """
    path_to_img: str

app = FastAPI()

@app.get("/")
def read_root():
    """
    Return a greeting. Used to test the API.

    Returns:
        dict: Greeting.
    """
    return {"Hello": "World"}

@app.get("/hello/{name}")
def hello_name(name: str):
    """
    Return a greeting for the specified name. Used to test the API.

    Args:
        name (str): Name to greet.

    Returns:
        dict: Greeting.
    """
    return {"Hello": name}

@app.post("/predict/")
def predict(image_path: ImagePath):
    """
    Predict a digit from an image using a trained MNIST model.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        dict: Predicted digit and confidence.
    """

    # Load the saved model
    model = load_model('../models/mnist_model')

    # Load and preprocess the input image
    image = load_and_preprocess_image(image_path.path_to_img)

    # Predict the digit
    predicted_digit, confidence = predict_digit(model, image)

    return {'prediction': int(predicted_digit), 'confidence': int(confidence*100)}

@app.post("/predict_file/")
def predict_file(image_file: UploadFile = File(...)):
    """
    Predict a digit from an image file using a trained MNIST model.

    Args:
        image_file (UploadFile): Image file to predict.

    Returns:
        dict: Predicted digit and confidence.
    """

    # Load the saved model
    model = load_model('../models/mnist_model')

    # Load and preprocess the input image
    image = load_and_preprocess_image(image_file.file)

    # Predict the digit
    predicted_digit, confidence = predict_digit(model, image)

    return {'prediction': int(predicted_digit), 'confidence': int(confidence*100)}



if __name__ == '__main__':
    """
    Run the API server.
    """

    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
