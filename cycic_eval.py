import pandas as pd
import argparse
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr
import numpy as np

def compute_accuracy(args):    
    preds = load_predictions(args.preds)
    labels = load_dataset_file(args.labels)    
    accuracy = accuracy_score(labels.correct_answer, preds)
    print("Dataset accuracy:", accuracy)

def load_dataset_file(filename):
    # @hack: assume True is answer index 0 and False is index 1
    df = pd.read_json(filename, lines=True, dtype={'correct_answer':bool})
    df.correct_answer = ~df.correct_answer
    return df

def load_predictions(filename):
    df = pd.read_csv(filename, names=['prediction'])
    if df.prediction.dtype == int:
        df.prediction = df.prediction.astype(bool)
        df.prediction = ~df.prediction
    else:
        df.prediction = df.prediction.astype(bool)
    return df

def main():
    parser = argparse.ArgumentParser(description="Evaluate prediction accuracy for entangled datasets")
    parser.add_argument("--labels", type=str, help="Labels (correct) for the origin data", required=True)
    parser.add_argument("--preds", type=str, help="Predictions for the origin data, one per line", required=True)
    
    args = parser.parse_args()
    compute_accuracy(args)

main()
