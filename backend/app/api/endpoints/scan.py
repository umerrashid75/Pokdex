from fastapi import APIRouter, File, UploadFile, HTTPException
from app.schemas.scan import ScanResponse
from app.services.ml_service import ml_service

router = APIRouter()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# ── Type mapping helpers ───────────────────────────────────────
ANIMAL_TYPE_MAP = {
    # predators / carnivores
    **{
        a: "predator"
        for a in [
            "lion",
            "tiger",
            "leopard",
            "cheetah",
            "jaguar",
            "wolf",
            "hyena",
            "eagle",
            "hawk",
            "falcon",
            "shark",
            "crocodile",
            "orca",
            "barracuda",
            "lynx",
            "coyote",
            "fox",
        ]
    },
    # mammals
    **{
        a: "mammal"
        for a in [
            "elephant",
            "gorilla",
            "orangutan",
            "chimpanzee",
            "bear",
            "panda",
            "koala",
            "kangaroo",
            "whale",
            "dolphin",
            "seal",
            "otter",
            "deer",
            "horse",
            "cow",
            "pig",
            "sheep",
            "goat",
            "dog",
            "cat",
            "rabbit",
            "rat",
            "mouse",
            "hamster",
            "squirrel",
            "raccoon",
            "bat",
            "hedgehog",
            "badger",
            "wombat",
            "possum",
            "reindeer",
            "bison",
            "donkey",
            "ox",
            "hippopotamus",
            "rhinoceros",
            "giraffe",
        ]
    },
    # bugs
    **{
        a: "bug"
        for a in [
            "bee",
            "beetle",
            "butterfly",
            "caterpillar",
            "cockroach",
            "dragonfly",
            "fly",
            "grasshopper",
            "ladybug",
            "mosquito",
            "moth",
            "ant",
            "wasp",
            "termite",
        ]
    },
    # aquatic
    **{
        a: "aquatic"
        for a in [
            "goldfish",
            "lobster",
            "crab",
            "jellyfish",
            "octopus",
            "seahorse",
            "squid",
            "starfish",
            "oyster",
            "penguin",
            "pelican",
            "duck",
            "goose",
            "flamingo",
            "swan",
        ]
    },
    # reptile
    **{
        a: "reptile"
        for a in [
            "lizard",
            "snake",
            "turtle",
            "crocodile",
            "gecko",
            "iguana",
            "chameleon",
        ]
    },
    # bird
    **{
        a: "bird"
        for a in [
            "eagle",
            "owl",
            "parrot",
            "pigeon",
            "sparrow",
            "crow",
            "woodpecker",
            "hummingbird",
            "hornbill",
            "sandpiper",
            "turkey",
            "pelican",
            "flamingo",
            "duck",
            "goose",
            "swan",
            "penguin",
        ]
    },
}


def _guess_type(name: str) -> str:
    return ANIMAL_TYPE_MAP.get(name.lower(), "normal")


@router.post("/scan", response_model=ScanResponse)
async def scan(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    try:
        data = await file.read()
    except Exception:
        raise HTTPException(400, "Could not read uploaded image file.")

    if not data:
        raise HTTPException(400, "Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image too large. Max size is 10MB.")

    try:
        clean_name, confidence = ml_service.predict(data)
    except Exception as e:
        raise HTTPException(
            400, f"Could not decode image or perform inference: {str(e)}"
        )

    # ── Lore generation ───────────────────────────────────────
    lore, fun_fact, lore_source = ml_service.generate_lore_and_fact(
        clean_name, confidence
    )
    animal_type = _guess_type(clean_name)

    return ScanResponse(
        name=clean_name,
        type=animal_type,
        confidence=confidence,
        lore=lore,
        fun_fact=fun_fact,
        lore_source=lore_source,
        model=ml_service.model_type,
    )
