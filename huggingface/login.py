"""
Hugging Face Hub Authentication Script

Features:
- Login using environment variable (HF_TOKEN)
- Falls back to interactive login if token not found
- Basic error handling
"""

import os
from huggingface_hub import login
from huggingface_hub.utils import HfHubHTTPError


def huggingface_login():
    try:
        token = os.getenv("HF_TOKEN")

        if token:
            print("[INFO] Using HF_TOKEN from environment variable...")
            login(token=token, add_to_git_credential=True)
            print("[SUCCESS] Logged in using environment token ✅")

        else:
            print("[INFO] HF_TOKEN not found.")
            print("[ACTION] Please login interactively...")
            login()
            print("[SUCCESS] Logged in interactively ✅")

    except HfHubHTTPError as e:
        print("[ERROR] Authentication failed ❌")
        print(f"Reason: {e}")

    except Exception as e:
        print("[ERROR] Unexpected error ❌")
        print(f"Details: {e}")


if __name__ == "__main__":
    huggingface_login()