from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load pipeline
model = joblib.load("yield_model.pkl")

@app.route('/predict', methods=['POST'])
def predict_yield():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        df = pd.DataFrame([data])  # Convert JSON to DataFrame
        prediction = model.predict(df)[0]

        return jsonify({'Predicted_Yield': float(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
