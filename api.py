from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# -------------------------
# 1. Load all models from models/ folder
# -------------------------
MODEL_DIR = "models"
models = {}

# Automatically load all .pkl files
for file in os.listdir(MODEL_DIR):
    if file.endswith(".pkl"):
        # Model name = filename without '_yield_model.pkl' or '.pkl'
        name = file.replace("_yield_model.pkl", "").replace(".pkl", "")
        models[name] = joblib.load(os.path.join(MODEL_DIR, file))

if not models:
    raise RuntimeError("❌ No model files found in 'models/' folder!")

print(f"✅ Loaded models: {list(models.keys())}")

# -------------------------
# 2. Prediction route
# -------------------------
@app.route('/predict', methods=['POST'])
def predict_yield():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Choose model (default = best if exists, else first available)
        model_name = data.get("model", "best" if "best" in models else list(models.keys())[0])
        if model_name not in models:
            return jsonify({
                "error": f"Invalid model '{model_name}'. Available models: {list(models.keys())}"
            }), 400

        # Prepare DataFrame (exclude "model" key)
        df = pd.DataFrame([{k: v for k, v in data.items() if k != "model"}])

        # Predict
        prediction = models[model_name].predict(df)[0]

        return jsonify({
            "model": model_name,
            "Predicted_Yield": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------
# 3. Run server
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
