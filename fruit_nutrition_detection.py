import requests
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from imageai.Classification import ImageClassification
from typing import Dict
import uvicorn
from ultralytics import YOLO

app = FastAPI()

USDA_API_KEY = '{USDA_API_KEY}'
USDA_BASE_URL = 'https://api.nal.usda.gov/fdc/v1/foods/search'

# Set up the ImageAI model
execution_path = os.getcwd()
prediction = ImageClassification()
# prediction.setModelTypeAsResNet50()
prediction.setModelPath(os.path.join(execution_path, "xception_weights_tf_dim_ordering_tf_kernels.h5"))
prediction.loadModel()


def fetch_nutritional_data(fruit_name: str) -> Dict:
    params = {
        'api_key': USDA_API_KEY,
        'query': fruit_name,
        'dataType': ['Survey (FNDDS)', 'Foundation'],
        'pageSize': 1
    }
    response = requests.get(USDA_BASE_URL, params=params)
    data = response.json()
    return data

def process_nutritional_data(data: Dict) -> Dict:
    if 'foods' not in data or len(data['foods']) == 0:
        return {"error": "No data found"}

    food_item = data['foods'][0]
    nutrients = food_item['foodNutrients']

    nutritional_info = {
        "name": food_item['description'],
        "nutrients": {nutrient['nutrientName']: nutrient['value'] for nutrient in nutrients}
    }

    return nutritional_info

@app.post("/fruit-detection", description="Fruit detection with nutrional infromation")
async def fruit_detection(file: UploadFile = File(...)):
    # Save the uploaded file
    file_location = f"{execution_path}/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(file.file.read())

    # Use the ImageAI model to classify the image
    predictions, probabilities = prediction.classifyImage(file_location, result_count=1)
    detected_fruit = predictions[0]

    # Fetch nutritional data for the detected fruit
    data = fetch_nutritional_data(detected_fruit)
    nutrition_info = process_nutritional_data(data)

    # Delete the uploaded file after processing
    os.remove(file_location)

    result = {
        "detected_fruit": detected_fruit,
        "probability": probabilities[0],
        "nutrition_info": nutrition_info
    }

    return JSONResponse(content=result)

if __name__ == "__main__":
  uvicorn.run(app, host='0.0.0.0', port=5000)
