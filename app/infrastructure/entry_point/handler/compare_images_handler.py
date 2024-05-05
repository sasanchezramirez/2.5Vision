from app.domain.usecase.compare_images import read_image

def compare_images_handler():
    img_1 = 'buena.jpg'
    img_2 = 'mala.jpg'
    message = {}
    message = read_image(img_1,img_2)
    return message