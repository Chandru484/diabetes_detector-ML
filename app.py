import joblib,sklearn
from flask import Flask, request, jsonify
import pandas as pd

# Load the model and scaler
model = joblib.load('logistic_regression_model.joblib')
scaler = joblib.load('standard_scaler.joblib')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Diabetes Prediction API! Use /predict endpoint to get predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        
        # Convert input data to DataFrame
        # Assuming input data is a dictionary where keys are feature names
        # and values are lists (for single prediction) or list of lists (for batch prediction)
        input_df = pd.DataFrame(data)
        
        # Scale the input features
        scaled_input = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)
        
        # Convert prediction to list and then to JSON serializable format
        results = []
        for i in range(len(prediction)):
            results.append({
                'prediction': int(prediction[i]),
                'probability_no_diabetes': float(prediction_proba[i][0]),
                'probability_diabetes': float(prediction_proba[i][1])
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
   app.run(debug=True)
 
# To run the app, uncomment the following line. 
# In a normal Python script, you would use `if __name__ == '__main__': app.run()`
# but for Colab with ngrok, `run_with_ngrok(app)` handles the execution.
# app.run()
