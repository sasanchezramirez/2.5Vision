from pydantic import BaseModel

class UploadImageResponse(BaseModel):
    image_url: str
    image_name: str
