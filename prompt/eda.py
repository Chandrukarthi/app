import pandas as pd
import os
from config import client
from prompts import get_eda_prompt

# File path
file_path = os.path.join(os.path.dirname(__file__), "data.csv")

def generate_eda_report(file_path):
    print("Loading data...")

    # Check file
    if not os.path.exists(file_path):
        raise FileNotFoundError("data.csv not found")

    df = pd.read_csv(file_path)

    print("Dataset Loaded Successfully")
    print("Shape:", df.shape)

    # Basic EDA
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Sample data for prompt
    sample_data = df.sample(min(15, len(df)), random_state=42).to_dict(orient='records')

    print("\nGenerating EDA Prompt...")
    prompt = get_eda_prompt(sample_data)

    print("Sending request to Gemini...")

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text


if __name__ == "__main__":
    try:
        report = generate_eda_report(file_path)

        print("\n--- EDA REPORT ---\n")
        print(report)
        print("\n------------------\n")

        # Save output
        with open("eda_report.txt", "w", encoding="utf-8") as f:
            f.write(report)

        print("Report saved as eda_report.txt")

    except Exception as e:
        print("Error:", e)