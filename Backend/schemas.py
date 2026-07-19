from enum import Enum
from pydantic import BaseModel


class Gender(str, Enum):
     MALE = "male"
     FEMALE = "female"

class PredictResponse(BaseModel):
    body_fat_percentage: float | None
    measurements: dict[str, float]
    body_fat_supported: bool
    category: str | None = None
    message: str | None = None