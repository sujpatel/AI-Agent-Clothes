from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["top", "bottom", "shoes", "outerwear", "accessory"]


class ClothingItem(BaseModel):
    category: Category
    color: str
    formality: int = Field(ge=1, le=5)  # 1 (very casual) - 5 (very formal)
    warmth: int = Field(ge=1, le=5)  # 1 (very light) - 5 (very warm)
    pattern: str
    description: str
