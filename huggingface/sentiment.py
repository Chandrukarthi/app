"""
Sentiment Analysis using HuggingFace Transformers
Optimized Version
"""

import argparse
from transformers import pipeline

print("[INFO] Loading model: distilbert-base-uncased-finetuned-sst-2-english...")
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def analyze_sentiment(texts):
    print("[INFO] Running sentiment analysis...")
    results = classifier(texts)
    return results

def display_results(texts, results):
    print("\n--- Sentiment Analysis Results ---")
    for text, result in zip(texts, results):
        print(f"Text : {text}")
        print(f"Label: {result['label']} (confidence: {result['score']:.4f})\n")

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