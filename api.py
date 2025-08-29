# api.py
import os
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# -------------------------
# 1. Load all models from models/ folder
# -------------------------
MODEL_DIR = "models"
models = {}

for file in os.listdir(MODEL_DIR):
    if file.endswith(".pkl"):
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

        model_name = data.get("model", "best" if "best" in models else list(models.keys())[0])
        if model_name not in models:
            return jsonify({
                "error": f"Invalid model '{model_name}'. Available: {list(models.keys())}"
            }), 400

        df = pd.DataFrame([{k: v for k, v in data.items() if k != "model"}])
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)