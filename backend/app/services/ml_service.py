import json
import logging
import re
from io import BytesIO
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from app.core.config import settings

log = logging.getLogger("pokedex.ml_service")

# ── Pre-processing (same for both models) ─────────────────────
PREPROCESS = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

ANIMAL_KEYWORDS = {
    "animal",
    "mammal",
    "bird",
    "fish",
    "reptile",
    "amphibian",
    "insect",
    "arachnid",
    "crustacean",
    "canine",
    "feline",
    "dog",
    "cat",
    "wolf",
    "fox",
    "bear",
    "lion",
    "tiger",
    "cheetah",
    "leopard",
    "jaguar",
    "horse",
    "deer",
    "cow",
    "goat",
    "sheep",
    "rabbit",
    "mouse",
    "rat",
    "squirrel",
    "bat",
    "elephant",
    "giraffe",
    "zebra",
    "rhino",
    "rhinoceros",
    "hippopotamus",
    "monkey",
    "ape",
    "gorilla",
    "chimpanzee",
    "orangutan",
    "kangaroo",
    "koala",
    "wombat",
    "otter",
    "raccoon",
    "badger",
    "hedgehog",
    "porcupine",
    "boar",
    "pig",
    "bison",
    "ox",
    "donkey",
    "panda",
    "whale",
    "dolphin",
    "seal",
    "shark",
    "ray",
    "stingray",
    "octopus",
    "squid",
    "lobster",
    "crab",
    "jellyfish",
    "starfish",
    "seahorse",
    "oyster",
    "penguin",
    "eagle",
    "hawk",
    "falcon",
    "owl",
    "parrot",
    "sparrow",
    "crow",
    "pigeon",
    "duck",
    "goose",
    "swan",
    "flamingo",
    "pelican",
    "woodpecker",
    "hummingbird",
    "toucan",
    "turkey",
    "ostrich",
    "hen",
    "rooster",
    "snake",
    "lizard",
    "gecko",
    "iguana",
    "chameleon",
    "crocodile",
    "alligator",
    "turtle",
    "frog",
    "toad",
    "bee",
    "beetle",
    "butterfly",
    "moth",
    "grasshopper",
    "dragonfly",
    "fly",
    "mosquito",
    "ant",
    "wasp",
    "termite",
    "cockroach",
    "spider",
    "scorpion",
}


class MLService:
    def __init__(self):
        self.net = None
        self.idx_to_class = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = "unknown"

        self.text_model = None
        self.text_tokenizer = None
        self.text_model_id = None
        self.static_lore = {}

    def load_models(self):
        self._load_classifier()
        self._load_static_lore()
        self._load_text_generator()
        log.info(
            "ML Service ready  [classifier=%s  lore=%s]",
            self.model_type,
            "textgen" if self.text_model else "static",
        )

    def _load_classifier(self) -> None:
        """Load fine-tuned ResNet50 if available, else stock ImageNet model."""
        if settings.CUSTOM_WEIGHTS.exists() and settings.CUSTOM_LABELS.exists():
            log.info(
                "Loading fine-tuned ResNet50 from %s ...", settings.CUSTOM_WEIGHTS.name
            )
            ckpt = torch.load(settings.CUSTOM_WEIGHTS, map_location=self.device)
            num_classes = ckpt["num_classes"]
            self.idx_to_class = {int(k): v for k, v in ckpt["class_labels"].items()}

            net = models.resnet50(weights=None)
            net.fc = torch.nn.Sequential(
                torch.nn.Dropout(0.45),
                torch.nn.Linear(net.fc.in_features, num_classes),
            )
            net.load_state_dict(ckpt["model_state"])
            net.to(self.device).eval()
            self.net = net
            self.model_type = "custom"
            log.info(
                "Fine-tuned model loaded (%d classes, val_acc=%.1f%%)",
                num_classes,
                ckpt.get("val_acc", 0) * 100,
            )
            return

        # ── Fallback: stock ImageNet ResNet50 ─────────────────────
        log.info("Loading stock ResNet50 (ImageNet fallback) ...")
        self.net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.net.to(self.device).eval()
        self.model_type = "imagenet"

        # Load human-readable ImageNet labels
        if settings.IMAGENET_LABELS_PATH.exists():
            with open(settings.IMAGENET_LABELS_PATH) as f:
                self.idx_to_class = {i: l.strip() for i, l in enumerate(f)}
        else:
            # Fetch from torchvision's built-in meta
            cat_data = models.ResNet50_Weights.IMAGENET1K_V2.meta.get("categories", {})
            self.idx_to_class = (
                {i: v["label"] for i, v in cat_data.items()}
                if hasattr(cat_data, "items")
                else {}
            )
            # Simple fallback
            if not self.idx_to_class and isinstance(cat_data, list):
                self.idx_to_class = {i: c for i, c in enumerate(cat_data)}

        log.info("ImageNet labels loaded (%d classes)", len(self.idx_to_class))

    def _load_text_generator(self) -> None:
        """Load a pretrained text model for animal description/facts."""
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            model_id = settings.NLP_MODEL_ID
            log.info("Loading NLP model %s ...", model_id)
            self.text_tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.text_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            self.text_model.eval()
            self.text_model_id = model_id
            log.info("NLP model loaded: %s", model_id)
        except Exception as e:
            log.warning("NLP model load failed (%s) - using static lore fallback", e)

    def _load_static_lore(self) -> None:
        if settings.SPECIES_DATA_PATH.exists():
            with open(settings.SPECIES_DATA_PATH) as f:
                entries = json.load(f)
            log.info("Static species DB loaded (%d entries)", len(entries))
            self.static_lore = {e["name"].lower(): e for e in entries.values()}

    def _lookup_static_entry(self, animal_name: str) -> Optional[dict]:
        entry_data = self.static_lore.get(animal_name.lower())
        if not entry_data:
            first = animal_name.lower().split()[0]
            entry_data = next(
                (v for k, v in self.static_lore.items() if first in k), None
            )
        return entry_data

    def _clean_label(self, raw_label: str) -> str:
        primary = raw_label.split(",")[0].strip()
        return primary.replace("_", " ").title()

    def _is_animal_label(self, raw_label: str) -> bool:
        label = raw_label.lower().replace("_", " ")
        tokens = re.findall(r"[a-z]+", label)
        return any(token in ANIMAL_KEYWORDS for token in tokens)

    def predict(self, image_bytes: bytes) -> tuple[str, float]:
        """Runs inference on the image and returns the predicted class name and confidence."""
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        tensor = PREPROCESS(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.net(tensor)
            probs = F.softmax(logits, dim=1)
            top_k = min(5, probs.shape[1])
            top_probs, top_idxs = probs[0].topk(top_k)

        picked_name = "Unknown"
        picked_conf = 0.0

        for conf, idx in zip(top_probs, top_idxs):
            raw_name = self.idx_to_class.get(int(idx.item()), "Unknown")
            if self.model_type == "custom" or self._is_animal_label(raw_name):
                picked_name = self._clean_label(raw_name)
                picked_conf = float(conf.item())
                break

        if picked_name == "Unknown":
            fallback_idx = int(top_idxs[0].item())
            fallback_name = self.idx_to_class.get(fallback_idx, "Unknown")
            picked_name = self._clean_label(fallback_name)
            picked_conf = float(top_probs[0].item())

        confidence = round(picked_conf * 100, 1)
        clean_name = picked_name
        log.info("Scanned -> %s (%.1f%%)", clean_name, confidence)
        return clean_name, confidence

    def generate_lore_and_fact(
        self, animal_name: str, confidence: float
    ) -> tuple[str, str, str]:
        """Return (description, fun_fact, source) for an identified animal."""
        static_entry = self._lookup_static_entry(animal_name)
        static_lore = static_entry.get("dex_entry") if static_entry else None

        if self.text_model and self.text_tokenizer:
            prompt = (
                "You are a Pokedex assistant. "
                "Given an animal name, write two concise lines:\n"
                "Description: one informative sentence.\n"
                "Fun Fact: one surprising factual sentence.\n"
                f"Animal: {animal_name}\n"
                f"Classifier confidence: {confidence:.1f}%\n"
                f"Reference text: {static_lore or 'No local reference available.'}"
            )
            enc = self.text_tokenizer(prompt, return_tensors="pt", truncation=True)
            with torch.no_grad():
                out = self.text_model.generate(
                    **enc,
                    max_new_tokens=settings.NLP_MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    repetition_penalty=1.08,
                )
            text = self.text_tokenizer.decode(out[0], skip_special_tokens=True).strip()

            desc_match = re.search(
                r"Description\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE
            )
            fact_match = re.search(
                r"Fun\s*Fact\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE
            )

            if desc_match and fact_match:
                lore = desc_match.group(1).strip()
                fun_fact = fact_match.group(1).strip()
                if lore and fun_fact:
                    return lore, fun_fact, "textgen"

            sentences = [
                s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()
            ]
            if len(sentences) >= 2:
                return sentences[0], sentences[1], "textgen"

        if static_lore:
            fallback_fact = (
                f"{animal_name} appears in the local Pokedex knowledge base."
            )
            return static_lore, fallback_fact, "static"

        lore = f"{animal_name} is an animal detected by the classifier."
        fun_fact = "This result uses a general-purpose vision model and may need human verification."
        return lore, fun_fact, "fallback"


# Create a singleton instance
ml_service = MLService()
