import argparse
from transformers import pipeline
from datasets import load_dataset

# Load models once
print("[INFO] Loading models...")
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

generation_model = pipeline(
    "text-generation",
    model="distilgpt2"
)

# ---------------- SENTIMENT ----------------
def run_sentiment_on_dataset(num_samples=5):
    print("\n[INFO] Loading dataset: imdb")
    dataset = load_dataset("imdb", split=f"test[:{num_samples}]")

    texts = [sample["text"][:200] + "..." for sample in dataset]

    print("\n--- Sentiment Analysis Results ---")
    results = sentiment_model(texts)

    for i, (text, result) in enumerate(zip(texts, results), 1):
        print(f"\n[{i}] Text : {text}")
        print(f"    Label: {result['label']} (confidence: {result['score']:.4f})")


# ---------------- GENERATION ----------------
def run_generation_on_dataset(num_samples=3):
    print("\n[INFO] Loading dataset: ag_news")
    dataset = load_dataset("ag_news", split=f"test[:{num_samples}]")

    prompts = [sample["text"][:50] + "..." for sample in dataset]

    print("\n--- Text Generation Results ---")

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}] Prompt : {prompt}")

        result = generation_model(
            prompt,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            pad_token_id=50256
        )

        print(f"    Generated: {result[0]['generated_text']}")


# ---------------- MAIN ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HuggingFace models on datasets")

    parser.add_argument(
        "--task",
        type=str,
        choices=["sentiment", "generation", "both"],
        default="both",
        help="Task to run"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of samples"
    )

    args = parser.parse_args()

    if args.task in ["sentiment", "both"]:
        run_sentiment_on_dataset(args.samples)

    if args.task in ["generation", "both"]:
        run_generation_on_dataset(args.samples)