from pydantic import BaseModel, Field

class ScanResponse(BaseModel):
    name: str = Field(..., description="Predicted name of the species")
    type: str = Field(..., description="Derived type category of the animal")
    confidence: float = Field(..., description="Confidence score percentage (0-100)")
    lore: str = Field(..., description="Pokédex lore descriptive text")
    lore_source: str = Field(..., description="Source of the lore: 'gpt2' or 'static'")
    model: str = Field(..., description="Model loaded: 'custom' or 'imagenet'")
