"""Command-line crop recommendation using the Kaggle crop dataset."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "crop_recommendation_dataset.csv"
DEFAULT_MODEL = ROOT / "models" / "crop_model.joblib"

# The linked Kaggle file uses N/P/K, temperature, humidity, ph, rainfall, label.
# The repository's bundled extension uses descriptive names and adds Soil/Carbon.
ALIASES = {
    "n": "Nitrogen", "nitrogen": "Nitrogen",
    "p": "Phosphorous", "phosphorus": "Phosphorous", "phosphorous": "Phosphorous",
    "k": "Potassium", "potassium": "Potassium",
    "temperature": "Temperature", "temp": "Temperature",
    "humidity": "Humidity", "relative_humidity": "Humidity",
    "ph": "PH", "soil_ph": "PH",
    "rainfall": "Rainfall", "rain": "Rainfall",
    "carbon": "Carbon", "soil_carbon": "Carbon",
    "soil": "Soil", "soil_type": "Soil",
    "label": "Crop", "crop": "Crop", "crop_name": "Crop",
}
FEATURE_ORDER = ["Nitrogen", "Phosphorous", "Potassium", "Temperature", "Humidity", "PH", "Rainfall", "Carbon", "Soil"]
NUMERIC_FEATURES = set(FEATURE_ORDER) - {"Soil"}


class CliError(Exception):
    """An expected input or model error that should be shown without a traceback."""


def _key(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def load_dataset(path: Path) -> tuple[pd.DataFrame, str, list[str]]:
    if not path.is_file():
        raise CliError(f"Dataset not found: {path}\nPass its location with --data PATH.")
    try:
        raw = pd.read_csv(path)
    except Exception as exc:
        raise CliError(f"Could not read CSV {path}: {exc}") from exc
    rename: dict[Any, str] = {}
    for col in raw.columns:
        alias = ALIASES.get(_key(col))
        if alias:
            rename[col] = alias
    data = raw.rename(columns=rename)
    if data.columns.duplicated().any():
        raise CliError("The CSV contains duplicate columns that map to the same feature name.")
    target = "Crop" if "Crop" in data.columns else None
    if target is None:
        raise CliError("Could not find the crop label column. Expected 'label' or 'Crop'.")
    features = [name for name in FEATURE_ORDER if name in data.columns]
    if not {"Nitrogen", "Phosphorous", "Potassium", "Temperature", "Humidity", "PH", "Rainfall"}.issubset(features):
        missing = sorted({"Nitrogen", "Phosphorous", "Potassium", "Temperature", "Humidity", "PH", "Rainfall"} - set(features))
        raise CliError("Dataset is missing required crop recommendation features: " + ", ".join(missing))
    data = data[features + [target]].copy()
    data[target] = data[target].astype("string").str.strip()
    data = data.dropna(subset=[target])
    if data[target].eq("").any():
        data = data.loc[~data[target].eq("")]
    if data.empty:
        raise CliError("The CSV has no rows with a crop label.")
    for feature in features:
        if feature in NUMERIC_FEATURES:
            data[feature] = pd.to_numeric(data[feature], errors="coerce")
        else:
            data[feature] = data[feature].astype("string").str.strip()
    invalid = data[features].isna().any(axis=1)
    if invalid.any():
        raise CliError(f"Found {int(invalid.sum())} rows with missing or invalid feature values. Clean the CSV and try again.")
    if data[target].nunique() < 2:
        raise CliError("Training requires at least two different crop labels.")
    return data, target, features


def _model_pipeline(features: list[str]) -> Pipeline:
    numeric = [feature for feature in features if feature in NUMERIC_FEATURES]
    categorical = [feature for feature in features if feature not in NUMERIC_FEATURES]
    transformers = []
    if numeric:
        transformers.append(("numeric", StandardScaler(), numeric))
    if categorical:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore"), categorical))
    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
    ])


def train(args: argparse.Namespace) -> None:
    data, target, features = load_dataset(Path(args.data).expanduser())
    counts = data[target].value_counts()
    can_holdout = len(data) >= 10 and counts.min() >= 2
    accuracy = None
    if can_holdout:
        x_train, x_test, y_train, y_test = train_test_split(
            data[features], data[target], test_size=args.test_size, random_state=args.seed,
            stratify=data[target],
        )
        evaluation_model = _model_pipeline(features)
        evaluation_model.fit(x_train, y_train)
        accuracy = accuracy_score(y_test, evaluation_model.predict(x_test))
    model = _model_pipeline(features)
    model.fit(data[features], data[target])
    destination = Path(args.model).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": model,
        "features": features,
        "classes": sorted(data[target].unique().tolist()),
        "rows": int(len(data)),
        "source": str(Path(args.data).expanduser().resolve()),
        "holdout_accuracy": accuracy,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    joblib.dump(artifact, temporary)
    temporary.replace(destination)
    print(f"Trained on {len(data):,} rows across {len(artifact['classes'])} crops.")
    print("Features: " + ", ".join(features))
    if accuracy is not None:
        print(f"Stratified holdout accuracy: {accuracy:.1%}")
    else:
        print("Holdout accuracy skipped (not enough samples per class).")
    print(f"Model saved to: {destination}")


def _read_model(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CliError(f"Model not found: {path}\nTrain one first: python cli.py train --data PATH_TO_CSV")
    try:
        artifact = joblib.load(path)
    except Exception as exc:
        raise CliError(f"Could not load model {path}: {exc}") from exc
    if not isinstance(artifact, dict) or "pipeline" not in artifact or "features" not in artifact:
        raise CliError("Model file has an unsupported format. Train a new model with this CLI.")
    return artifact


def _prediction_row(values: dict[str, Any], features: list[str]) -> pd.DataFrame:
    normalized: dict[str, Any] = {}
    for key, value in values.items():
        feature = ALIASES.get(_key(key))
        if feature:
            normalized[feature] = value
    missing = [feature for feature in features if feature not in normalized]
    if missing:
        raise CliError("Missing prediction values: " + ", ".join(missing))
    row = {}
    for feature in features:
        value = normalized[feature]
        if feature in NUMERIC_FEATURES:
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise CliError(f"{feature} must be a number; received {value!r}.") from exc
            if not np.isfinite(value):
                raise CliError(f"{feature} must be a finite number.")
        else:
            value = str(value).strip()
            if not value:
                raise CliError(f"{feature} cannot be empty.")
        row[feature] = value
    return pd.DataFrame([row], columns=features)


def _parse_values(items: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise CliError(f"Expected FEATURE=VALUE, received {item!r}.")
        key, value = item.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def predict(args: argparse.Namespace) -> None:
    artifact = _read_model(Path(args.model).expanduser())
    features = artifact["features"]
    values = _parse_values(args.values or [])
    if args.interactive:
        for feature in features:
            if feature not in values:
                entered = input(f"{feature}: ").strip()
                values[feature] = entered
    row = _prediction_row(values, features)
    prediction = artifact["pipeline"].predict(row)[0]
    print(f"Recommended crop: {prediction}")


def info(args: argparse.Namespace) -> None:
    data, target, features = load_dataset(Path(args.data).expanduser())
    print(f"Dataset: {Path(args.data).expanduser()}")
    print(f"Rows: {len(data):,}")
    print(f"Crop classes: {data[target].nunique()}")
    print("Features: " + ", ".join(features))
    print("Crops: " + ", ".join(sorted(data[target].unique().tolist())))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and use a crop recommendation model from a CSV dataset.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="Train a model from a CSV")
    train_parser.add_argument("--data", default=str(DEFAULT_DATA), help=f"CSV path (default: {DEFAULT_DATA.name})")
    train_parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Output model path")
    train_parser.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction used to report accuracy (default: 0.2)")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.set_defaults(func=train)
    predict_parser = subparsers.add_parser("predict", help="Predict a crop from feature values")
    predict_parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Trained model path")
    predict_parser.add_argument("--values", nargs="*", metavar="FEATURE=VALUE", help="Input values, e.g. N=90 P=42")
    predict_parser.add_argument("--interactive", action="store_true", help="Prompt for any missing model features")
    predict_parser.set_defaults(func=predict)
    info_parser = subparsers.add_parser("info", help="Show the detected dataset schema and crop classes")
    info_parser.add_argument("--data", default=str(DEFAULT_DATA), help=f"CSV path (default: {DEFAULT_DATA.name})")
    info_parser.set_defaults(func=info)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "test_size", 0.2) <= 0 or getattr(args, "test_size", 0.2) >= 1:
        parser.error("--test-size must be between 0 and 1")
    try:
        args.func(args)
    except CliError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
