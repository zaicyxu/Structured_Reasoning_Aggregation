# config.py

class Config:
    # Base API endpoint
    BASE_URL = "https://api.chatanywhere.tech/v1"

    # API keys for different models
    API_KEYS = {
        "gpt": "YOUR_API_KEY",
        "gemini": "YOUR_API_KEY",
        "qwen": "YOUR_API_KEY",
        "deepseek": "YOUR_API_KEY",
    }

    # Model mapping
    MODELS = {
        "gpt": "gpt-3.5-turbo",
        "gemini": "gpt-3.5-turbo",
        "qwen": "gpt-3.5-turbo",
        "deepseek": "gpt-3.5-turbo",
    }

    # Generation parameters
    TEMPERATURE = 0.7
    MAX_TOKENS = 800