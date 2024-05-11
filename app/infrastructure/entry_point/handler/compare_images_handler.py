from app.domain.usecase.compare_images import read_image

def compare_images_handler():
    """
    Handles the comparison between two images, retrieving the brightest and the sharpest images.

    Returns:
        dict: A dictionary containing keys 'Brightest_img' and 'Sharptest_img' with their respective values.
    """
    img_1 = 'buena_fixed.png'
    img_2 = 'mala_fixed.png'
    img_3 = 'peldar.jpeg'
    message = []
    message = read_image(img_2,img_3)
    response= {
        "Brightest_img": message[0],
        "Sharptest_img": message[1]
    }
    return response