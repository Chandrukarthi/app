"""
Optimized Text Generation using HuggingFace Transformers
"""

import argparse
from transformers import pipeline

# Load model once
print("[INFO] Loading model: distilgpt2...")
generator = pipeline("text-generation", model="distilgpt2")

def generate_text(prompt, max_tokens=30, temperature=0.7):
    print(f"[INFO] Generating text for prompt: '{prompt}'")

    result = generator(
        prompt,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=50256   # avoids warning
    )

    return result[0]["generated_text"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using distilgpt2")

    parser.add_argument("--prompt", type=str, default="AI will change")
    parser.add_argument("--max_tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.7)

    args = parser.parse_args()

    output = generate_text(args.prompt, args.max_tokens, args.temperature)

    print("\n--- Generated Text ---")
    print(output)