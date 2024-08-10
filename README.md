# FruitNutritionDetector

**FruitNutritionDetector** is a powerful **FastAPI-based API** designed to detect fruits from images and retrieve their nutritional information using **ImageAI** and the **USDA API**. This API utilizes advanced models like Xception for image classification and fetches nutritional data from the USDA FoodData Central database.

## Key Features

- **Fruit Detection API:** Detect fruits from images with high accuracy using the ImageAI framework.
- **Nutritional Information Retrieval:** Fetch detailed nutritional data from the USDA API for the detected fruit.
- **FastAPI Framework:** Built with FastAPI, providing high performance and easy deployment.
- **YOLO Integration:** Potential for advanced object detection with YOLO models.

## Getting Started

### Installation

Follow these steps to set up the **FruitNutritionDetector** API:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/FruitNutritionDetector.git
   cd FruitNutritionDetector
   
2. **Install dependencies:**
   
   Make sure you have Python 3.7+ installed, then install the required packages:
   ```bash
   pip install fastapi requests pyngrok uvicorn
   pip install cython 'pillow>=7.0.0' 'numpy>=1.18.1' 'opencv-python>=4.1.2' 'torch>=1.9.0' --extra-index-url 'https://download.pytorch.org/whl/cpu' 'torchvision>=0.10.0' --extra-index-url     'https://download.pytorch.org/whl/cpu' 'pytest==7.1.3' 'tqdm==4.64.1' 'scipy>=1.7.3' 'matplotlib>=3.4.3' 'mock==4.0.3'
   
3. **Download pre-trained models:**
   ```bash
   wget https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/resnet50-19c8e357.pth/ -O resnet50-19c8e357.pth
   wget https://github.com/vishalbharti1990/deep_feature_clustering/blob/master/Documents/.keras/models/xception_weights_tf_dim_ordering_tf_kernels.h5?raw=true -O xception_weights_tf_dim_ordering_tf_kernels.h5

4. **Set up USDA API Key:**
   
   Replace {USDA_API_KEY} in the code with your actual USDA API key.

5. **Launch the API:**

   Start the server with Uvicorn:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 5000
   The API will be accessible at http://localhost:5000.


6. **API Usage**
   
   Endpoint: POST /fruit-detection
   This endpoint detects a fruit from an uploaded image and retrieves its nutritional information.

   Request Parameters:

   - file: The image file containing the fruit (Supported formats: .jpg, .png, .jpeg).
     ```bash
     curl -X POST "http://localhost:5000/fruit-detection" \
       -H "accept: application/json" \
       -H "Content-Type: multipart/form-data" \
       -F "file=@/path/to/fruit_image.jpg"

   Response:
   - detected_fruit: The name of the detected fruit.
   - probability: The probability score of the detection.
   - nutrition_info: A dictionary containing the nutritional information retrieved from the USDA API.
     ```bash
       {
        "detected_fruit": "Apple",
        "probability": 97.5,
        "nutrition_info": {
          "name": "Apple",
          "nutrients": {
            "Energy": 52,
            "Protein": 0.26,
            "Total lipid (fat)": 0.17,
            "Carbohydrate, by difference": 13.81
          }
        }
      }

## Contributing
  We welcome contributions to FruitNutritionDetector! Feel free to fork the repository, create a branch, and submit a Pull Request with your improvements or new features.

