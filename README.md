# Rice Leaf Diseases Prediction App

This web application allows users to predict rice leaf diseases using a pre-trained deep learning model. Users can either upload an image or take a picture using their device's camera. Additionally, the app displays real-time data fetched from Firebase Realtime Database.

## How to Use

### 1. Upload Image

- Click on "Choose an option" and select "Upload Image."
- Upload an image in JPG, JPEG, or PNG format.
- Click the "Proceed" button to classify the uploaded image.
- The app will display the uploaded image along with the predicted class and confidence score.

### 2. Take a Picture

- Click on "Choose an option" and select "Take a picture."
- Use your device's camera to capture an image.
- Click the "Proceed" button to classify the captured image.
- The app will display the captured image along with the predicted class and confidence score.

### Real-time Data from Firebase (IOT + Firebase)

- The app also displays real-time data fetched from Firebase Realtime Database.
- Any changes in the database will be reflected in the app in real-time.

## Requirements

- Python 3.x
- Streamlit
- Pillow (PIL)
- NumPy
- Keras
- Firebase Admin SDK

## Setup

1. Install the required libraries by running:

   ```bash
   pip install -r requirements.txt
2. Run the app:

     ```bash
   streamlit run app.py

## Model and Labels

## .env ( Add fireabase realtime db url )

The app utilizes a pre-trained deep learning model for rice leaf disease prediction. The model file is located at `./model/rice_leaf_diseases.h5`, and the labels are stored in `./model/diseases_labels.txt`.
