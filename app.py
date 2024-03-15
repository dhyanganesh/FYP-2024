# import pandas as pd
# from flask import Flask, request, jsonify
# import joblib
# import os

# app = Flask(__name__)

# # Define the path to the trained model
# MODEL_PATH = 'random_forest_model.pkl'

# # Check if the model file exists
# if not os.path.exists(MODEL_PATH):
#     print(f"Error: Model file '{MODEL_PATH}' not found.")
#     exit(1)

# # Load trained model
# try:
#     with open(MODEL_PATH, 'rb') as f:
#         model = joblib.load(f)
#     print("Model loaded successfully.")
# except Exception as e:
#     print("Error loading the model:", e)  # Print full error message
#     exit(1)

# @app.route('/')
# def index():
#     return app.send_static_file('index.html')
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get input data
#         data = request.json
#         # Log input data received
#         print('Input data received:', data)
#         # Preprocess input data (convert to DataFrame, perform any necessary transformations)
#         # Convert input data to DataFrame
#         input_df = pd.DataFrame([data])
        
#         # Perform feature reduction (select relevant columns)
#         selected_features = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes']
#         input_df = input_df[selected_features]
        
#         # Perform prediction
#         prediction = model.predict(input_df)
#         # Log prediction result
#         print('Prediction result:', prediction)
#         # Return prediction
#         return jsonify({'prediction': prediction[0]})
#     except Exception as e:
#         # Log any errors that occur during prediction
#         print('Prediction error:', e)
#         return jsonify({'error': str(e)})

###########################################################################################

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
