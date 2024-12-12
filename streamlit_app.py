import pandas as pd
from PIL import Image
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model as tf_load_model  # For CNN models

# Function to load a CNN model
def load_model(model_name: str):
    """
    Load a pre-trained CNN model.

    Args:
        model_name (str): Name of the model file to load.

    Returns:
        The loaded model.
    """
    try:
        model = tf_load_model(model_name)  # Load TensorFlow/Keras model
        print(f"Loaded CNN Model: {model_name}")
        print(f"Model Input Shape: {model.input_shape}")  # Check model input shape
        return model
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")
        raise

# Function to preprocess and classify an image
def classify_image(img: bytes, model) -> pd.DataFrame:
    """
    Classify the given image using the selected CNN model and return predictions.

    Args:
        img (bytes): The image file to classify.
        model: The pre-trained CNN model.

    Returns:
        pd.DataFrame: A DataFrame containing predictions and their probabilities.
        str: The top predicted class.
    """
    try:
        # Load and preprocess the image
        image = Image.open(img).convert("RGB")
        input_shape = model.input_shape[1:3]  # Get height and width from model input shape
        image_resized = image.resize(input_shape)  # Resize image to model's expected size
        image_array = np.array(image_resized) / 255.0  # Normalize pixel values
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict probabilities
        probabilities = model.predict(image_array)[0]

        # Dynamically map class labels based on the number of probabilities
        num_classes = len(probabilities)
        if num_classes == 2:  # Binary classification
            class_labels = ["Not Fractured", "Fractured"]
        else:  # Multi-class (extendable for more classes)
            class_labels = [f"Class {i}" for i in range(num_classes)]

        # Create a DataFrame to store predictions and probabilities
        prediction_df = pd.DataFrame({
            "Class": class_labels,
            "Probability": probabilities
        }).sort_values("Probability", ascending=False)

        # Get the top prediction
        top_prediction = prediction_df.iloc[0]["Class"]

        return prediction_df, top_prediction

    except Exception as e:
        st.error(f"An error occurred during classification: {e}")
        return pd.DataFrame(), None

# Streamlit app
st.title("Bone Structure Analysis")
st.write("Upload an X-ray or bone scan image to analyze the structure.")

# Upload image
image_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Model selection
model_type = st.selectbox("Choose a model:", ["CNN (with Dropout)", "CNN (without Dropout)"])

# Load the selected model
try:
    model_files = {
        "CNN (with Dropout)": "small_cnn_with_dropout.h5",
        "CNN (without Dropout)": "sma;;_cnn_without_dropout.h5"
    }
    selected_model_file = model_files.get(model_type)
    if not selected_model_file:
        st.error(f"Model type {model_type} is not recognized.")
        st.stop()

    # Load the CNN model
    model = load_model(selected_model_file)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

if image_file:
    st.image(image_file, caption="Uploaded Image", use_column_width=True)
    pred_button = st.button("Analyze Bone Structure")
    
    if pred_button:
        # Perform image classification
        predictions_df, top_prediction = classify_image(image_file, model)

        if not predictions_df.empty:
            # Display top prediction
            st.success(f'Predicted Structure: **{top_prediction}** '
                       f'Confidence: {predictions_df.iloc[0]["Probability"]:.2%}')

            # Display all predictions
            st.write("Detailed Predictions:")
            st.table(predictions_df)
        else:
            st.error("Failed to classify the image.")
