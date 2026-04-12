import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.router import api_router
from app.services.ml_service import ml_service

log = logging.getLogger("pokedex")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize and load models
    ml_service.load_models()
    yield
    log.info("Shutting down...")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "status": "online",
        "classifier": ml_service.model_type,
        "lore": "gpt2" if ml_service.gpt2 else "static",
    }

app.include_router(api_router)
