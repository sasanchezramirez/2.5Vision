from fastapi import APIRouter
from app.infrastructure.entry_point.handler.hello_world_handler import create_message
from app.infrastructure.entry_point.handler.compare_images_handler import compare_images_handler
from app.infrastructure.entry_point.dto.hello_world_dto import HelloWorldRequest

router = APIRouter()

@router.post("/hello-world")
async def read_hello_world(bodyRequest: HelloWorldRequest):
    return create_message(bodyRequest.name)

@router.get("/compare-images", response_model=dict, summary="Compare two images",
            description="Compares two predetermined images and identifies the one with greater brightness and sharper edges.")
async def read_compare_images():
    """
    Endpoint to compare two images and return the results as JSON.

    This function wraps the `compare_images_handler` which processes the actual image comparison.

    Returns:
        dict: JSON object with the brightest and sharpest images' information.
    """
    return compare_images_handler()