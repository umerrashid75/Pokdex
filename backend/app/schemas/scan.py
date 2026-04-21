from pydantic import BaseModel, Field


class ScanResponse(BaseModel):
    name: str = Field(..., description="Predicted name of the species")
    type: str = Field(..., description="Derived type category of the animal")
    confidence: float = Field(..., description="Confidence score percentage (0-100)")
    lore: str = Field(..., description="Pokédex lore descriptive text")
    fun_fact: str = Field(
        ..., description="A concise interesting fact about the animal"
    )
    lore_source: str = Field(
        ..., description="Source of the lore: 'textgen', 'static', or 'fallback'"
    )
    model: str = Field(..., description="Model loaded: 'custom' or 'imagenet'")
