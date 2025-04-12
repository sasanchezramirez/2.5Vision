from pydantic import BaseModel

class UploadImageResponse(BaseModel):
    image_url: str
    image_name: str
    image_size: int
    image_type: str