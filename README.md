# Crop Recommendation System

This project recommends a crop from the [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/siddharthss/crop-recommendation-dataset?resource=download). The checked-in `crop_recommendation_dataset.csv` is that dataset: 2,200 rows, 22 crop labels, and seven input features (`N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`).

The CLI and web API train/use a model from this CSV. The React dashboard also uses crop profiles computed from the same rows. No `Carbon` or `Soil` columns are present in this dataset.

## Install

From the repository root:

```sh
python -m pip install -r requirements.txt
```

## CLI

Inspect the dataset and train the model:

```sh
python cli.py info
python cli.py train
```

The default dataset is `crop_recommendation_dataset.csv`; pass `--data PATH` to use another compatible CSV. The trained pipeline is saved to `models/crop_model.joblib`.

Predict by answering prompts:

```sh
python cli.py predict --interactive
```

Or pass all seven features directly:

```sh
python cli.py predict --values N=90 P=42 K=43 temperature=20.9 humidity=82 ph=6.5 rainfall=203
```

Use `python cli.py --help` for command options. Training prints a stratified holdout score, then fits the saved model on the full dataset. This score measures performance on a random split of this dataset and does not guarantee field performance.

## Web app

Train the model first, then start the Flask API:

```sh
python cli.py train
python app.py
```

In another terminal, start the React dashboard:

```sh
cd frontend
npm install
npm run dev
```

Open the local Vite URL printed in the terminal. The dashboard sends the same seven numeric inputs to the Flask API. Its crop profiles and demo scenarios are derived from the checked-in Kaggle CSV.

## Dataset columns

| CLI input | CSV column |
| --- | --- |
| Nitrogen | `N` |
| Phosphorus | `P` |
| Potassium | `K` |
| Temperature (°C) | `temperature` |
| Relative humidity (%) | `humidity` |
| Soil pH | `ph` |
| Rainfall (mm) | `rainfall` |
| Target crop | `label` |
