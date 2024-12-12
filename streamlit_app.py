import joblib
import pandas as pd
from PIL import Image
import streamlit as st
import numpy as np
import cv2  # For SIFT feature extraction
from tensorflow.keras.models import load_model  # For loading CNN models
from tensorflow.keras.preprocessing.image import img_to_array

# Function to load a model
def load_model_file(model_name: str):
    """
    Load a pre-trained model.

    Args:
        model_name (str): Name of the model file to load.

    Returns:
        sklearn.base.BaseEstimator or keras.Model: The loaded model.
    """
    if model_name.endswith(".pkl"):
        return joblib.load(model_name)
    elif model_name.endswith(".h5"):
        return load_model(model_name)  # Load CNN models
    else:
        raise ValueError("Unsupported model file format")

# Function to extract SIFT features
def extract_features(img) -> np.ndarray:
    """
    Extract features from the image using SIFT.

    Args:
        img (PIL.Image): The input image.

    Returns:
        np.ndarray: Feature vector of fixed size (128).
    """
    image_cv = np.array(img)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)  # Convert to grayscale

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image_cv, None)

    if descriptors is not None:
        return descriptors.flatten()[:128]  # Truncate/pad to fixed size
    else:
        return np.zeros(128)  # Zero vector if no features are found

# Function to preprocess and classify an image
def classify_image(img: bytes, model, model_type: str) -> pd.DataFrame:
    """
    Classify the given image using the selected model and return predictions.

    Args:
        img (bytes): The image file to classify.
        model: The pre-trained model.
        model_type (str): The type of model (KNN, ANN, SVM, or CNN).

    Returns:
        pd.DataFrame: A DataFrame containing predictions and their probabilities.
    """
    try:
        image = Image.open(img).convert("RGB")

        if model_type in ["KNN", "ANN", "SVM"]:
            # Extract features for non-CNN models
            features = extract_features(image)
            prediction = model.predict([features])
            probabilities = model.predict_proba([features])[0]
        elif model_type == "CNN":
            # Preprocess image for CNN
            image = image.resize((128, 128))  # Resize to match CNN input size
            image_array = img_to_array(image) / 255.0  # Normalize to [0, 1]
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
            probabilities = model.predict(image_array)[0]
            prediction = [1 if probabilities > 0.5 else 0]

        # Map numeric predictions to descriptive labels
        LABEL_MAPPING = {
            0: "Not Fractured",
            1: "Fractured"
        }
        class_labels = [LABEL_MAPPING[i] for i in range(len(probabilities))]

        # Create a DataFrame to store predictions and probabilities
        prediction_df = pd.DataFrame({
            "Class": class_labels,
            "Probability": probabilities
        })
        return prediction_df.sort_values("Probability", ascending=False), LABEL_MAPPING[prediction[0]]

    except Exception as e:
        st.error(f"An error occurred during classification: {e}")
        return pd.DataFrame(), None

# Streamlit app
st.title("Bone Structure Analysis")
st.write("Upload an X-ray or bone scan image to analyze the structure.")

# Upload image
image_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Model selection
model_type = st.selectbox("Choose a model:", ["KNN", "ANN", "SVM", "CNN"])

# Load the selected model
try:
    model_files = {
        "KNN": "knn_classifier.pkl",
        "ANN": "ann_classifier.pkl",
        "SVM": "svm_classifier.pkl",
        "CNN": "cnn_with_dropoutt.h5"  # Use the trained CNN model with dropout
    }
    selected_model_file = model_files[model_type]
    model = load_model_file(selected_model_file)
except FileNotFoundError as e:
    st.error(f"Missing file: {e}")
    st.stop()

if image_file:
    st.image(image_file, caption="Uploaded Image", use_column_width=True)
    pred_button = st.button("Analyze Bone Structure")
    
    if pred_button:
        # Perform image classification
        predictions_df, top_prediction = classify_image(image_file, model, model_type)

        if not predictions_df.empty:
            # Display top prediction
            st.success(f'Predicted Structure: **{top_prediction}** '
                       f'Confidence: {predictions_df.iloc[0]["Probability"]:.2%}')

            # Display all predictions
            st.write("Detailed Predictions:")
            st.table(predictions_df)
        else:
            st.error("Failed to classify the image.")
