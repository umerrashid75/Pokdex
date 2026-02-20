"""
backend/main.py  –  Smart Pokédex API
======================================
POST /v1/scan  – accept an image, return species + Pokédex lore

Model loading priority (checked on startup):
  1. pokedex_classifier.pth + class_labels.json   → your fine-tuned ResNet50
  2. pokedex_gpt2/                                → your fine-tuned GPT-2 (lore)
  3. Fallback: stock ResNet50 (ImageNet) + static species_data.json lore
"""

import json, logging, re
from contextlib import asynccontextmanager
from pathlib import Path
from io import BytesIO

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

log = logging.getLogger("pokedex")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR             = Path(__file__).parent
CUSTOM_WEIGHTS       = BASE_DIR / "pokedex_classifier.pth"
CUSTOM_LABELS        = BASE_DIR / "class_labels.json"
GPT2_DIR             = BASE_DIR / "pokedex_gpt2"
SPECIES_DATA_PATH    = BASE_DIR / "species_data.json"
IMAGENET_LABELS_PATH = BASE_DIR / "imagenet_classes.txt"

# Shared state (populated in lifespan)
state: dict = {}

# ── Pre-processing (same for both models) ─────────────────────
PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


def _load_classifier() -> tuple:
    """Load fine-tuned ResNet50 if available, else stock ImageNet model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if CUSTOM_WEIGHTS.exists() and CUSTOM_LABELS.exists():
        log.info("⚡ Loading fine-tuned ResNet50 from %s …", CUSTOM_WEIGHTS.name)
        ckpt         = torch.load(CUSTOM_WEIGHTS, map_location=device)
        num_classes  = ckpt["num_classes"]
        idx_to_class = {int(k): v for k, v in ckpt["class_labels"].items()}

        net = models.resnet50(weights=None)
        net.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.45),
            torch.nn.Linear(net.fc.in_features, num_classes),
        )
        net.load_state_dict(ckpt["model_state"])
        net.to(device).eval()
        log.info("✅ Fine-tuned model loaded (%d classes, val_acc=%.1f%%)",
                 num_classes, ckpt.get("val_acc", 0) * 100)
        return net, idx_to_class, device, "custom"

    # ── Fallback: stock ImageNet ResNet50 ─────────────────────
    log.info("⚡ Loading stock ResNet50 (ImageNet fallback) …")
    net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    net.to(device).eval()

    # Load human-readable ImageNet labels
    if IMAGENET_LABELS_PATH.exists():
        with open(IMAGENET_LABELS_PATH) as f:
            idx_to_class = {i: l.strip() for i, l in enumerate(f)}
    else:
        # Fetch from torchvision's built-in meta
        idx_to_class = {i: v["label"] for i, v in
                        models.ResNet50_Weights.IMAGENET1K_V2.meta["categories"].items()
                        if isinstance(i, int)} if hasattr(
            models.ResNet50_Weights.IMAGENET1K_V2.meta.get("categories", {}), "items"
        ) else {}
        # Simple fallback
        if not idx_to_class:
            cats = models.ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
            idx_to_class = {i: c for i, c in enumerate(cats)}

    log.info("✅ ImageNet labels loaded (%d classes)", len(idx_to_class))
    return net, idx_to_class, device, "imagenet"


def _load_gpt2():
    """Load fine-tuned GPT-2 if available."""
    if not GPT2_DIR.exists():
        return None, None
    try:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        log.info("⚡ Loading fine-tuned GPT-2 from %s …", GPT2_DIR)
        tokenizer = GPT2TokenizerFast.from_pretrained(str(GPT2_DIR))
        model     = GPT2LMHeadModel.from_pretrained(str(GPT2_DIR))
        model.eval()
        log.info("✅ GPT-2 lore model loaded")
        return model, tokenizer
    except Exception as e:
        log.warning("GPT-2 load failed (%s) – using static lore fallback", e)
        return None, None


def _load_static_lore() -> dict:
    if SPECIES_DATA_PATH.exists():
        with open(SPECIES_DATA_PATH) as f:
            entries = json.load(f)
        log.info("📖 Static species DB loaded (%d entries)", len(entries))
        return {e["name"].lower(): e for e in entries}
    return {}


# ── GPT-2 lore generation ──────────────────────────────────────
def _generate_lore(animal_name: str, gpt2_model, tokenizer) -> str:
    prompt = f"<ANIMAL>{animal_name}</ANIMAL><ENTRY>"
    enc    = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = gpt2_model.generate(
            **enc,
            max_new_tokens     = 120,
            do_sample          = True,
            temperature        = 0.82,
            top_p              = 0.92,
            repetition_penalty = 1.15,
            pad_token_id       = tokenizer.eos_token_id,
        )
    text  = tokenizer.decode(out[0], skip_special_tokens=True)
    entry = text.split("<ENTRY>")[-1].split("</ENTRY>")[0].strip()
    # Clean up any trailing partial sentence
    sentences = re.split(r'(?<=[.!?])\s', entry)
    cleaned   = " ".join(sentences[:-1]) if len(sentences) > 1 else entry
    return cleaned or entry


# ── Lifespan ───────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    net, idx_to_class, device, model_type = _load_classifier()
    gpt2_model, tokenizer                 = _load_gpt2()
    static_lore                           = _load_static_lore()

    state.update({
        "net":          net,
        "idx_to_class": idx_to_class,
        "device":       device,
        "model_type":   model_type,
        "gpt2":         gpt2_model,
        "tokenizer":    tokenizer,
        "static_lore":  static_lore,
    })
    log.info("🟢 Pokédex ready  [classifier=%s  lore=%s]",
             model_type, "gpt2" if gpt2_model else "static")
    yield
    log.info("🔴 Shutting down …")
    state.clear()


# ── App ────────────────────────────────────────────────────────
app = FastAPI(title="Smart Pokédex API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── Type mapping helpers ───────────────────────────────────────
ANIMAL_TYPE_MAP = {
    # predators / carnivores
    **{a: "predator" for a in ["lion","tiger","leopard","cheetah","jaguar","wolf",
                               "hyena","eagle","hawk","falcon","shark","crocodile",
                               "orca","barracuda","lynx","coyote","fox"]},
    # mammals
    **{a: "mammal"   for a in ["elephant","gorilla","orangutan","chimpanzee","bear",
                               "panda","koala","kangaroo","whale","dolphin","seal",
                               "otter","deer","horse","cow","pig","sheep","goat",
                               "dog","cat","rabbit","rat","mouse","hamster",
                               "squirrel","raccoon","bat","hedgehog","badger",
                               "wombat","possum","reindeer","bison","donkey","ox",
                               "hippopotamus","rhinoceros","giraffe"]},
    # bugs
    **{a: "bug"      for a in ["bee","beetle","butterfly","caterpillar","cockroach",
                               "dragonfly","fly","grasshopper","ladybugs","mosquito",
                               "moth","porcupine","ant","wasp","termite"]},
    # aquatic
    **{a: "aquatic"  for a in ["goldfish","lobster","crab","jellyfish","octopus",
                               "seahorse","squid","starfish","oyster","penguin",
                               "pelican","duck","goose","flamingo","swan"]},
    # reptile
    **{a: "reptile"  for a in ["lizard","snake","turtle","crocodile","gecko",
                               "iguana","chameleon"]},
    # bird
    **{a: "bird"     for a in ["eagle","owl","parrot","pigeon","sparrow","crow",
                               "woodpecker","hummingbird","hornbill","sandpiper",
                               "turkey","pelecan","flamingo","duck","goose","swan","penguin"]},
}

def _guess_type(name: str) -> str:
    return ANIMAL_TYPE_MAP.get(name.lower(), "normal")


@app.get("/")
async def root():
    return {
        "status":     "online",
        "classifier": state.get("model_type", "unknown"),
        "lore":       "gpt2" if state.get("gpt2") else "static",
    }


@app.post("/v1/scan")
async def scan(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    try:
        data  = await file.read()
        img   = Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image")

    # ── Inference ─────────────────────────────────────────────
    tensor = PREPROCESS(img).unsqueeze(0).to(state["device"])
    with torch.no_grad():
        logits = state["net"](tensor)
        probs  = F.softmax(logits, dim=1)
        conf, idx = probs[0].max(0)

    raw_name   = state["idx_to_class"].get(idx.item(), "Unknown")
    # Capitalise and clean ImageNet-style names (e.g. "tabby_cat" → "Tabby Cat")
    clean_name = raw_name.replace("_", " ").title()
    confidence = round(conf.item() * 100, 1)

    log.info("🔍 Scanned → %s (%.1f%%)", clean_name, confidence)

    # ── Lore generation ───────────────────────────────────────
    gpt2      = state.get("gpt2")
    tokenizer = state.get("tokenizer")

    if gpt2 and tokenizer:
        lore = _generate_lore(clean_name, gpt2, tokenizer)
        lore_source = "gpt2"
    else:
        # Static JSON fallback
        static = state.get("static_lore", {})
        entry  = static.get(clean_name.lower())
        if not entry:
            # Fuzzy: first word match
            first = clean_name.lower().split()[0]
            entry = next((v for k, v in static.items() if first in k), None)
        if entry:
            lore = entry.get("description", "Data not found in Pokédex.")
        else:
            lore = (f"{clean_name} is a fascinating creature whose full data entry "
                    f"has yet to be catalogued. Train the GPT-2 model for richer entries.")
        lore_source = "static"

    animal_type = _guess_type(clean_name)

    return {
        "name":       clean_name,
        "type":       animal_type,
        "confidence": confidence,
        "lore":       lore,
        "lore_source": lore_source,
        "model":      state.get("model_type", "unknown"),
    }
