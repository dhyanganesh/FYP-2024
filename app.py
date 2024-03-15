from flask import Flask, request, jsonify, render_template
import joblib
import os
import pandas as pd

app = Flask(__name__)

# Define the path to the trained model
MODEL_PATH = 'random_forest_model_reduced.pkl'

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found.")
    exit(1)

# Load trained model
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading the model:", e)
    exit(1)

@app.route('/')
def index():
    # Render the HTML file
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        data = request.json

        # Preprocess input data
        input_df = pd.DataFrame([data])

        # Perform prediction
        prediction = model.predict(input_df)

        # Return prediction
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
