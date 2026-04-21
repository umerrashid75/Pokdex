import os
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "Smart Pokédex API"
    VERSION: str = "2.0.0"

    # API
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Base Paths (Relative to app/core/config.py -> app/..)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

    # Model Paths
    CUSTOM_WEIGHTS: Path = BASE_DIR / "pokedex_classifier.pth"
    CUSTOM_LABELS: Path = BASE_DIR / "class_labels.json"
    # NLP model for generated description/fun fact
    NLP_MODEL_ID: str = "google/flan-t5-small"
    NLP_MAX_NEW_TOKENS: int = 96

    # Data Paths
    SPECIES_DATA_PATH: Path = BASE_DIR / "app" / "data" / "species_data.json"
    IMAGENET_LABELS_PATH: Path = BASE_DIR / "imagenet_classes.txt"

    def cors_origin_list(self) -> List[str]:
        return [
            origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()
        ]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
