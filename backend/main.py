"""
Smart Pokedex – FastAPI Backend
================================
POST /v1/scan  →  Accepts an image, runs ResNet50 inference,
                  returns {label, confidence, type, dex_entry}.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
import torchvision.models as models
import torchvision.transforms as T
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pokedex")

# ---------------------------------------------------------------------------
# ImageNet class labels (1000 classes)
# torchvision ships with a helper that downloads them once.
# We fall back to downloading via urllib if the helper isn't available.
# ---------------------------------------------------------------------------
def _load_imagenet_labels() -> list[str]:
    """Return the 1000 ImageNet class name strings."""
    try:
        from torchvision.models._api import get_model_weights  # noqa: F401  (just a probe)
    except ImportError:
        pass

    # Use the well-known synset list bundled with torchvision examples
    import urllib.request, urllib.error  # noqa: E401

    url = (
        "https://raw.githubusercontent.com/pytorch/hub/master/"
        "imagenet_classes.txt"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return [line.strip() for line in resp.read().decode().splitlines()]
    except urllib.error.URLError:
        pass

    # Hard fallback – first 1000 indices named numerically so the app
    # still runs offline (lore lookup will use the generic template).
    return [f"class_{i}" for i in range(1000)]


# ---------------------------------------------------------------------------
# App state (model + data loaded once at startup)
# ---------------------------------------------------------------------------
class AppState:
    model: torch.nn.Module
    labels: list[str]
    species_db: dict[str, Any]
    transform: T.Compose


state = AppState()

SPECIES_DB_PATH = Path(__file__).parent / "species_data.json"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model + data on startup; clean up on shutdown."""
    log.info("⚡ Loading ResNet-50 …")
    state.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    state.model.eval()

    state.labels = _load_imagenet_labels()
    log.info(f"✅ ImageNet labels loaded ({len(state.labels)} classes)")

    with SPECIES_DB_PATH.open() as f:
        state.species_db = json.load(f)
    log.info(f"📖 Species DB loaded ({len(state.species_db)} entries)")

    state.transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    yield  # ← server runs here

    log.info("🔴 Shutting down …")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Smart Pokedex API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _clean_label(raw: str) -> str:
    """
    ImageNet labels look like 'n02099601 golden_retriever'.
    Return just 'golden_retriever'.
    """
    parts = raw.split(" ", 1)
    return parts[1] if len(parts) == 2 else raw


def _lookup_lore(clean_label: str) -> dict[str, str]:
    """
    Try to find a lore entry. The DB keys use underscores and lower-case.
    We try several normalized variants before giving up.
    """
    variants = [
        clean_label,
        clean_label.lower(),
        clean_label.replace("-", "_"),
    ]
    for v in variants:
        if v in state.species_db:
            entry = state.species_db[v]
            return {
                "name": entry["name"],
                "type": entry["type"],
                "dex_entry": entry["dex_entry"],
            }

    # Generic fallback
    display_name = clean_label.replace("_", " ").title()
    return {
        "name": display_name,
        "type": "Unknown",
        "dex_entry": (
            f"A remarkable creature known as the {display_name}. "
            "Little is known about this species in the Pokedex archives. "
            "Field researchers continue to study its behaviour and habitat."
        ),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
async def root():
    return {"status": "online", "message": "Smart Pokedex API is running."}


@app.post("/v1/scan", tags=["Inference"])
async def scan(file: UploadFile = File(...)):
    """
    Accept an image file, run ResNet-50 inference, and return Pokedex data.

    Returns
    -------
    {
        "label": "golden_retriever",
        "display_name": "Golden Retriever",
        "type": "Mammal",
        "confidence": 0.94,
        "dex_entry": "..."
    }
    """
    # --- Validate content type ---
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Send an image.",
        )

    # --- Read & decode ---
    raw_bytes = await file.read()
    try:
        img = Image.open(BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {exc}")

    # --- Preprocess ---
    tensor = state.transform(img).unsqueeze(0)  # [1, 3, 224, 224]

    # --- Inference ---
    with torch.no_grad():
        logits = state.model(tensor)             # [1, 1000]
        probs  = torch.softmax(logits, dim=1)
        top_prob, top_idx = probs.topk(1, dim=1)

    raw_label = state.labels[top_idx.item()]
    label     = _clean_label(raw_label)
    confidence = round(float(top_prob.item()), 4)

    lore = _lookup_lore(label)

    log.info(f"🔍 Scanned → {label} ({confidence*100:.1f}%)")

    return {
        "label":        label,
        "display_name": lore["name"],
        "type":         lore["type"],
        "confidence":   confidence,
        "dex_entry":    lore["dex_entry"],
    }
