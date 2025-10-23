# app/models.py
from pydantic import BaseModel
from typing import List

class PreprocessingResponse(BaseModel):
    transcribed_text: str
    base64_frames: List[str]
    noise_level: float
    
    class Config:
        schema_extra = {
            "example": {
                "transcribed_text": "Welcome to this beautiful three-bedroom home...",
                "base64_frames": [
                    "/9j/4AAQSk... (base64 string 1)",
                    "/9j/4AAQSk... (base64 string 2)"
                ],
                "noise_level": 0.0023
            }
        }