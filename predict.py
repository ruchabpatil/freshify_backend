import base64
import io
import os

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from ai_edge_litert.interpreter import Interpreter


app = Flask(__name__)
CORS(app)


# --------------------------------------------------
# Load TFLite model
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "multi_fruit_mlp_baseline.tflite"
)

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_image(image_bytes):

    # Open image
    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    # Resize to model input size
    image = image.resize((32, 32))

    # Convert to NumPy
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Normalize
    image_array = image_array / 255.0

    # Flatten
    image_flattened = image_array.reshape(
        1, 3072
    )

    # Send image to TFLite model
    interpreter.set_tensor(
        input_details[0]["index"],
        image_flattened
    )

    # Run inference
    interpreter.invoke()

    # Get prediction
    prediction = interpreter.get_tensor(
        output_details[0]["index"]
    )[0][0]

    prediction = float(prediction)

    # Classification
    predicted_class = (
        "Rotten"
        if prediction >= 0.5
        else "Fresh"
    )

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

        # Remove data URL prefix if present
        if "," in image_data:
            image_data = image_data.split(
                ",",
                1
            )[1]

        # Decode image
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
