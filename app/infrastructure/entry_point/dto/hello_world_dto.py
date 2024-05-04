from pydantic import BaseModel

class HelloWorldRequest(BaseModel):
    name: str
    id: str