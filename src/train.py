import argparse
import os
import sys

from joblib import dump
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline

def load_and_validate_data(data_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV and ensures it has the required columns.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    try:
        df = pd.read_csv(data_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV file is empty: {data_path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"CSV file is malformed: {data_path}") from exc
    except OSError as exc:
        raise OSError(f"Could not read data file: {data_path}") from exc

    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns")
    if df.empty:
        raise ValueError("CSV contains no rows")
    if df["text"].isna().all():
        raise ValueError("'text' column cannot be entirely empty")
    if df["label"].isna().all():
        raise ValueError("'label' column cannot be entirely empty")
    return df

def split_data(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Splits the DataFrame into training and testing sets.
    """
    try:
        # Stratified split is preferred
        X_train, X_test, y_train, y_test = train_test_split(
            df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
        )
    except ValueError:
        # Fallback if stratification fails (e.g., on very small datasets)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                df["text"], df["label"], test_size=0.2, random_state=42
            )
        except ValueError as exc:
            raise ValueError(
                "Could not split dataset; ensure at least 2 valid rows are present"
            ) from exc
    return X_train, X_test, y_train, y_test

def print_dataset_summary(df: pd.DataFrame) -> None:
    """
    Print a compact overview of dataset quality and class balance.

    This is intended as a quick pre-training sanity check so users can see
    whether the dataset has missing values and how labels are distributed.
    """
    total_rows = len(df)
    missing_text = int(df["text"].isna().sum())
    missing_label = int(df["label"].isna().sum())
    # Keep missing labels visible in the distribution output as well.
    label_counts = df["label"].value_counts(dropna=False).to_dict()

    print("Dataset summary:")
    print(f"- Rows: {total_rows}")
    print(f"- Missing text values: {missing_text}")
    print(f"- Missing label values: {missing_label}")
    print(f"- Label distribution: {label_counts}")

def train_model(X_train: pd.Series, y_train: pd.Series) -> Pipeline:
    """
    Builds and trains a classification pipeline.
    """
    clf_pipeline = make_pipeline(
        TfidfVectorizer(min_df=1, ngram_range=(1, 2)),
        LogisticRegression(max_iter=1000),
    )
    try:
        clf_pipeline.fit(X_train, y_train)
    except ValueError as exc:
        raise ValueError(
            "Model training failed; check that text and label values are valid"
        ) from exc
    return clf_pipeline

def main(data_path: str, model_path: str) -> None:
    """
    Main workflow to load, train, evaluate, and save the model.
    """
    try:
        df = load_and_validate_data(data_path)
        print_dataset_summary(df)
        X_train, X_test, y_train, y_test = split_data(df)
        clf = train_model(X_train, y_train)

        # Evaluate and print accuracy
        acc = clf.score(X_test, y_test)
        print(f"Test accuracy: {acc:.3f}")

        save_model(clf, model_path)
    except Exception as exc:
        print(f"Training pipeline failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

def save_model(model: Pipeline, model_path: str) -> None:
    """
    Saves the trained model to a file.
    """
    output_dir = os.path.dirname(model_path) or "."
    try:
        os.makedirs(output_dir, exist_ok=True)
        dump(model, model_path)
    except OSError as exc:
        raise OSError(f"Failed to save model to {model_path}") from exc
    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sentiments.csv")
    parser.add_argument("--out", default="models/sentiment.joblib")

    args: argparse.Namespace = parser.parse_args()
    main(data_path=args.data, model_path=args.out)
