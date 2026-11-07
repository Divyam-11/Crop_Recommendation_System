# Crop Recommendation Dashboard

This Vite/React frontend uses the Kaggle dataset profiles generated from the repository's `crop_recommendation_dataset.csv` and sends predictions to the Flask API.

From the repository root, install Python dependencies and train the model first:

```sh
python -m pip install -r requirements.txt
python cli.py train
python app.py
```

In another terminal:

```sh
npm install
npm run dev
```

The prediction form uses the dataset's seven numeric fields: nitrogen, phosphorus, potassium, temperature, humidity, soil pH, and rainfall. See the root [README](../README.md) for CLI usage and CSV details.
