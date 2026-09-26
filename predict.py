import base64
import io
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS


# --------------------------------------------------
# Flask application
# --------------------------------------------------

app = Flask(__name__)

# Allow requests from the Vercel frontend
CORS(app)


# --------------------------------------------------
# Load model
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "multi_fruit_mlp_baseline.keras"
)

model = tf.keras.models.load_model(MODEL_PATH)


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_image(image_bytes):

    # Open image and convert to RGB
    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    # Resize to the same size used during training
    image = image.resize((32, 32))

    # Convert image to NumPy array
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Normalize pixel values
    image_array = image_array / 255.0

    # Flatten image
    image_flattened = image_array.reshape(
        1,
        3072
    )

    # Model prediction
    prediction = model.predict(
        image_flattened,
        verbose=0
    )[0][0]

    prediction = float(prediction)

    # Interpret prediction
    if prediction >= 0.5:
        predicted_class = "Rotten"
    else:
        predicted_class = "Fresh"

    fresh_score = 1.0 - prediction
    rotten_score = prediction

    return {
        "prediction": predicted_class,
        "fresh_score": round(
            fresh_score * 100,
            2
        ),
        "rotten_score": round(
            rotten_score * 100,
            2
        )
    }


# --------------------------------------------------
# API endpoint
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({
                "error": "No image provided."
            }), 400

        image_data = data["image"]

        # Remove data URL prefix
        if "," in image_data:
            image_data = image_data.split(
                ",",
                1
            )[1]

        # Decode Base64 image
        image_bytes = base64.b64decode(
            image_data
        )

        # Predict
        result = predict_image(
            image_bytes
        )

        return jsonify(result)

    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 500


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Freshify API is running!",
        "status": "online"
    })


# --------------------------------------------------
# Local development
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
