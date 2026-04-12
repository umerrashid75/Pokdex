import json
import logging
import re
from io import BytesIO

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from app.core.config import settings

log = logging.getLogger("pokedex.ml_service")

# ── Pre-processing (same for both models) ─────────────────────
PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class MLService:
    def __init__(self):
        self.net = None
        self.idx_to_class = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = "unknown"
        
        self.gpt2 = None
        self.tokenizer = None
        self.static_lore = {}

    def load_models(self):
        self._load_classifier()
        self._load_gpt2()
        self._load_static_lore()
        log.info("ML Service ready  [classifier=%s  lore=%s]",
                 self.model_type, "gpt2" if self.gpt2 else "static")

    def _load_classifier(self) -> None:
        """Load fine-tuned ResNet50 if available, else stock ImageNet model."""
        if settings.CUSTOM_WEIGHTS.exists() and settings.CUSTOM_LABELS.exists():
            log.info("Loading fine-tuned ResNet50 from %s ...", settings.CUSTOM_WEIGHTS.name)
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
            log.info("Fine-tuned model loaded (%d classes, val_acc=%.1f%%)",
                     num_classes, ckpt.get("val_acc", 0) * 100)
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
            self.idx_to_class = {i: v["label"] for i, v in cat_data.items()} if hasattr(cat_data, "items") else {}
            # Simple fallback
            if not self.idx_to_class and isinstance(cat_data, list):
                self.idx_to_class = {i: c for i, c in enumerate(cat_data)}

        log.info("ImageNet labels loaded (%d classes)", len(self.idx_to_class))

    def _load_gpt2(self) -> None:
        """Load fine-tuned GPT-2 if available."""
        if not settings.GPT2_DIR.exists():
            return
        try:
            from transformers import GPT2LMHeadModel, GPT2TokenizerFast
            log.info("Loading fine-tuned GPT-2 from %s ...", settings.GPT2_DIR)
            self.tokenizer = GPT2TokenizerFast.from_pretrained(str(settings.GPT2_DIR))
            self.gpt2 = GPT2LMHeadModel.from_pretrained(str(settings.GPT2_DIR))
            self.gpt2.eval()
            log.info("GPT-2 lore model loaded")
        except Exception as e:
            log.warning("GPT-2 load failed (%s) – using static lore fallback", e)

    def _load_static_lore(self) -> None:
        if settings.SPECIES_DATA_PATH.exists():
            with open(settings.SPECIES_DATA_PATH) as f:
                entries = json.load(f)
            log.info("Static species DB loaded (%d entries)", len(entries))
            self.static_lore = {e["name"].lower(): e for e in entries.values()}

    def predict(self, image_bytes: bytes) -> tuple[str, float]:
        """Runs inference on the image and returns the predicted class name and confidence."""
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        tensor = PREPROCESS(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.net(tensor)
            probs = F.softmax(logits, dim=1)
            conf, idx = probs[0].max(0)

        raw_name = self.idx_to_class.get(idx.item(), "Unknown")
        # Capitalise and clean ImageNet-style names (e.g. "tabby_cat" → "Tabby Cat")
        clean_name = raw_name.replace("_", " ").title()
        confidence = round(conf.item() * 100, 1)
        log.info("Scanned -> %s (%.1f%%)", clean_name, confidence)
        return clean_name, confidence

    def generate_lore(self, animal_name: str) -> tuple[str, str]:
        """Returns the lore and the source (gpt2/static) for the given animal_name."""
        if self.gpt2 and self.tokenizer:
            prompt = f"<ANIMAL>{animal_name}</ANIMAL><ENTRY>"
            enc = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                out = self.gpt2.generate(
                    **enc,
                    max_new_tokens=120,
                    do_sample=True,
                    temperature=0.82,
                    top_p=0.92,
                    repetition_penalty=1.15,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            entry = text.split("<ENTRY>")[-1].split("</ENTRY>")[0].strip()
            # Clean up any trailing partial sentence
            sentences = re.split(r'(?<=[.!?])\s', entry)
            lore = " ".join(sentences[:-1]) if len(sentences) > 1 else entry
            lore = lore or entry
            return lore, "gpt2"

        # Static JSON fallback
        entry_data = self.static_lore.get(animal_name.lower())
        if not entry_data:
            # Fuzzy: first word match
            first = animal_name.lower().split()[0]
            entry_data = next((v for k, v in self.static_lore.items() if first in k), None)
            
        if entry_data:
            lore = entry_data.get("dex_entry", "Data not found in Pokédex.")
        else:
            lore = (f"{animal_name} is a fascinating creature whose full data entry "
                    f"has yet to be catalogued. Train the GPT-2 model for richer entries.")
        return lore, "static"

# Create a singleton instance
ml_service = MLService()
