import joblib
import pandas as pd
from PIL import Image
import streamlit as st
import numpy as np
import cv2  # For SIFT feature extraction
from tensorflow.keras.models import load_model as tf_load_model  # For CNN models

# Function to load a model
def load_model(model_name: str, is_cnn: bool = False):
    """
    Load a pre-trained model.

    Args:
        model_name (str): Name of the model file to load.
        is_cnn (bool): Indicates if the model is a CNN.

    Returns:
        The loaded model.
    """
    try:
        if is_cnn:
            model = tf_load_model(model_name)  # Load TensorFlow/Keras model
            print(f"Loaded CNN Model: {model_name}")
            print(f"Model Input Shape: {model.input_shape}")  # Check model input shape
            return model
        else:
            model = joblib.load(model_name)  # Load traditional ML model
            print(f"Loaded Traditional ML Model: {model_name}")
            return model
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")
        raise

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
        features = None

        if model_type in ["KNN", "ANN", "SVM"]:
            features = extract_features(image)
            prediction = model.predict([features])
            probabilities = model.predict_proba([features])[0]
        elif "CNN" in model_type:
            # Preprocess image for CNN
            input_shape = model.input_shape[1:3]  # Get height and width from model input shape
            image_resized = image.resize(input_shape)  # Resize image to model's expected size
            image_array = np.array(image_resized) / 255.0  # Normalize pixel values
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

            probabilities = model.predict(image_array)[0]  # Predict probabilities
            prediction = np.argmax(probabilities)  # Get class with highest probability
        else:
            st.error("Unsupported model type.")
            return pd.DataFrame(), None

        # Map numeric predictions to descriptive labels
        LABEL_MAPPING = {
            0: "Not Fractured",
            1: "Fractured"
        }
        class_labels = list(LABEL_MAPPING.values())

        # Create a DataFrame to store predictions and probabilities
        prediction_df = pd.DataFrame({
            "Class": class_labels,
            "Probability": probabilities
        })

        return prediction_df.sort_values("Probability", ascending=False), LABEL_MAPPING[prediction]

    except Exception as e:
        st.error(f"An error occurred during classification: {e}")
        return pd.DataFrame(), None

# Streamlit app
st.title("Bone Structure Analysis")
st.write("Upload an X-ray or bone scan image to analyze the structure.")

# Upload image
image_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Model selection
model_type = st.selectbox(
    "Choose a model:", 
    ["KNN", "ANN", "SVM", "CNN (with Dropout)", "CNN (without Dropout)"]
)

# Load the selected model
try:
    model_files = {
        "KNN": "knn_classifier.pkl",
        "ANN": "ann_classifier.pkl",
        "SVM": "svm_classifier.pkl",
        "CNN (with Dropout)": "cnn_with_dropoutt.h5",
        "CNN (without Dropout)": "cnn_without_dropoutt.h5"
    }
    selected_model_file = model_files.get(model_type)
    if not selected_model_file:
        st.error(f"Model type {model_type} is not recognized.")
        st.stop()

    is_cnn = "CNN" in model_type  # Determine if the selected model is a CNN
    model = load_model(selected_model_file, is_cnn=is_cnn)
except Exception as e:
    st.error(f"Error loading model: {e}")
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
