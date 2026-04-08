"""
HuggingFace Hub Authentication (Improved)
"""

import os
from huggingface_hub import login

def huggingface_login():
    try:
        token = os.environ.get("HF_TOKEN")

        if token:
            print("[INFO] Using HF_TOKEN from environment variable...")
            login(token=token)
            print("[SUCCESS] Logged in using environment token ✅")
        else:
            print("[INFO] No HF_TOKEN found.")
            print("[ACTION] Starting interactive login...")
            login()
            print("[SUCCESS] Logged in interactively ✅")

    except Exception as e:
        print("[ERROR] Login failed ❌")
        print(f"Details: {e}")

if __name__ == "__main__":
    huggingface_login()