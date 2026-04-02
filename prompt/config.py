from google import genai
import os
from dotenv import load_dotenv

# Load .env from current folder
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

api_key = os.getenv("GEMINI_API_KEY")

print("Loaded API Key:", api_key)  # DEBUG

if not api_key:
    raise ValueError("API key not found")

client = genai.Client(api_key=api_key)