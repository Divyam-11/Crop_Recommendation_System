"""Flask API for the Kaggle crop recommendation model."""

from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib

from cli import CliError, DEFAULT_MODEL, _prediction_row


app = Flask(__name__)
CORS(app)


def load_model():
    try:
        artifact = joblib.load(DEFAULT_MODEL)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Model missing at {DEFAULT_MODEL}. Run 'python cli.py train' first."
        ) from exc
    if not isinstance(artifact, dict) or "pipeline" not in artifact or "features" not in artifact:
        raise RuntimeError(f"Unsupported model artifact: {DEFAULT_MODEL}")
    return artifact


@app.get("/health")
def health():
    try:
        artifact = load_model()
    except RuntimeError as exc:
        return jsonify({"status": "error", "model_loaded": False, "error": str(exc)}), 503
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "features": artifact["features"],
        "crops": len(artifact.get("classes", [])),
        "rows": artifact.get("rows"),
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send input values as a JSON object."}), 400
    try:
        artifact = load_model()
        row = _prediction_row(data, artifact["features"])
        result = artifact["pipeline"].predict(row)[0]
    except CliError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    return jsonify({"predicted_crop": str(result)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
