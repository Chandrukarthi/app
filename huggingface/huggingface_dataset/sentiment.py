"""
Optimized Sentiment Analysis using HuggingFace Transformers
"""

import argparse
from transformers import pipeline

# Load model once
print("[INFO] Loading model: distilbert-base-uncased-finetuned-sst-2-english...")
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def analyze_sentiment(texts):
    print("[INFO] Running sentiment analysis...")
    return classifier(texts)

def display_results(texts, results):
    print("\n--- Sentiment Analysis Results ---")

    for i, (text, result) in enumerate(zip(texts, results), 1):
        label = result["label"]
        score = result["score"]

        emoji = "😊" if label == "POSITIVE" else "😠"

        print(f"\n[{i}] Text : {text}")
        print(f"    Label: {label} {emoji}")
        print(f"    Confidence: {score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze sentiment of text")

    parser.add_argument(
        "--texts",
        nargs="+",
        default=["This product is amazing!"],
        help="Enter one or more sentences"
    )

    args = parser.parse_args()

    results = analyze_sentiment(args.texts)
    display_results(args.texts, results)