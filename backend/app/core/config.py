import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "Smart Pokédex API"
    VERSION: str = "2.0.0"

    # Base Paths (Relative to app/core/config.py -> app/..)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    
    # Model Paths
    CUSTOM_WEIGHTS: Path = BASE_DIR / "pokedex_classifier.pth"
    CUSTOM_LABELS: Path = BASE_DIR / "class_labels.json"
    GPT2_DIR: Path = BASE_DIR / "pokedex_gpt2"
    
    # Data Paths
    SPECIES_DATA_PATH: Path = BASE_DIR / "app" / "data" / "species_data.json"
    IMAGENET_LABELS_PATH: Path = BASE_DIR / "imagenet_classes.txt"

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
