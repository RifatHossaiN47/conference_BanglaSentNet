"""
BanglaSentNet Inference & Evaluation CLI
=========================================
Runs aspect-based sentiment analysis on user-specified Bangla review text
or evaluates benchmark batches from sample datasets.

Usage:
    # Run evaluation on the sample dataset:
    python src/inference.py

    # Run inference on a custom Bangla review:
    python src/inference.py --text "প্রোডাক্টের কোয়ালিটি চমৎকার কিন্তু ডেলিভারি পেতে ৭ দিন লাগলো"
"""

import os
import sys
import argparse
import pandas as pd

# Safe path resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import clean_bangla_text, get_review_length_category
from src.models import (
    DynamicEnsembleClassifier,
    rule_guided_feature_extractor,
    ASPECTS,
    SENTIMENTS
)


def run_single_inference(text: str, verbose: bool = True):
    """
    Performs end-to-end multi-aspect sentiment prediction on a single review text.
    """
    cleaned = clean_bangla_text(text)
    length_cat = get_review_length_category(cleaned)
    
    classifier = DynamicEnsembleClassifier()
    weights = classifier.get_weights_for_text(cleaned)
    raw_preds = rule_guided_feature_extractor(cleaned)
    aggregated = classifier.aggregate_predictions(cleaned, raw_preds)
    
    if verbose:
        print("\n" + "=" * 70)
        print(" BanglaSentNet: Multi-Aspect Sentiment Analysis Result")
        print("=" * 70)
        print(f" Raw Text:       {text}")
        print(f" Cleaned Text:   {cleaned}")
        print(f" Length Group:   {length_cat.upper()} (BanglaBERT: {weights['BanglaBERT']:.2f}, BiLSTM: {weights['BiLSTM']:.2f}, LSTM: {weights['LSTM']:.2f}, GRU: {weights['GRU']:.2f})")
        print("-" * 70)
        print(f" {'ASPECT':<15} | {'PREDICTED SENTIMENT':<20} | {'CONFIDENCE':<12}")
        print("-" * 70)
        for aspect in ASPECTS:
            pred = aggregated[aspect]
            sentiment_tag = pred['predicted_sentiment']
            conf = pred['confidence']
            print(f" {aspect:<15} | {sentiment_tag:<20} | {conf:<12.2%}")
        print("=" * 70 + "\n")
        
    return aggregated


def run_batch_evaluation(csv_path: str = "data/sample_reviews.csv"):
    """
    Runs batch evaluation on the sample reviews CSV and displays a summary table.
    """
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    print("\n" + "=" * 95)
    print(f" BanglaSentNet Batch Evaluation on {len(df)} Sample E-Commerce Reviews")
    print("=" * 95)
    print(f" {'ID':<3} | {'Platform':<9} | {'Quality':<10} | {'Service':<10} | {'Price':<10} | {'Decoration':<10} | {'Review Snippet':<32}")
    print("-" * 95)
    
    classifier = DynamicEnsembleClassifier()
    for _, row in df.iterrows():
        review = str(row['review_text'])
        cleaned = clean_bangla_text(review)
        raw_preds = rule_guided_feature_extractor(cleaned)
        preds = classifier.aggregate_predictions(cleaned, raw_preds)
        
        q = preds['Quality']['predicted_sentiment']
        s = preds['Service']['predicted_sentiment']
        p = preds['Price']['predicted_sentiment']
        d = preds['Decoration']['predicted_sentiment']
        
        snippet = review[:30] + "..." if len(review) > 30 else review
        print(f" {row['review_id']:<3} | {row['platform']:<9} | {q:<10} | {s:<10} | {p:<10} | {d:<10} | {snippet:<32}")
        
    print("=" * 95 + "\n")


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        
    parser = argparse.ArgumentParser(description="BanglaSentNet Aspect Sentiment Analysis")
    parser.add_argument("--text", type=str, default=None, help="Bangla review text to analyze")
    parser.add_argument("--csv", type=str, default="data/sample_reviews.csv", help="CSV path for batch evaluation")
    args = parser.parse_args()
    
    if args.text:
        run_single_inference(args.text)
    else:
        run_batch_evaluation(args.csv)


if __name__ == "__main__":
    main()
