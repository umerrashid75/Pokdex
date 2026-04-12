from fastapi import APIRouter
from app.api.endpoints import scan

api_router = APIRouter()
api_router.include_router(scan.router, prefix="/v1", tags=["Scan"])
