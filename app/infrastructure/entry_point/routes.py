from fastapi import APIRouter
from app.infrastructure.entry_point.handler.hello_world_handler import create_message
from app.infrastructure.entry_point.dto.hello_world_dto import HelloWorldRequest

router = APIRouter()

@router.post("/hello-world")
async def read_hello_world(bodyRequest: HelloWorldRequest):
    return create_message(bodyRequest.name)