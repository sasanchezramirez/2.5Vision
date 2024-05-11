import cv2
import numpy as np

def read_image(img_1, img_2):
    """
    Execute  method for this usecase. It reads two images and returns 
    the image with better weather conditions
    """
    img_1 = cv2.imread(img_1)
    img_2 = cv2.imread(img_2)
    img_1_processed = cv2.cvtColor(np.array(img_1), cv2.COLOR_RGB2BGR)
    img_2_processed = cv2.cvtColor(np.array(img_2), cv2.COLOR_RGB2BGR)
    message = compare_image(img_1_processed, img_2_processed)
    print (f"Retorna {message}")
    return message['brightest_image'], message['image_with_most_edges']

def compare_image(img_1, img_2):
    """
    This function handles the images and execute the especific methods for 
    visual analyzing
    """
    results = {}
    images = {
    'Image 1': img_1,
    'Image 2': img_2
    }
    results['brightest_image'] = compare_brightness(images)
    results['image_with_most_edges'] = compare_edges(images)
    return results

def compare_brightness(images):
    """
    This function returns the name of the brightest image.
    :param images: Dict of image names and their respective image data.
    :return: Name of the brightest image.
    """
    max_brightness = 0  # This value would be the maximum value obteined from an image
    brightest_image = None  

    for name,img in images.items():
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  
        brightness = cv2.mean(hsv)[2] 

        if brightness > max_brightness:
            max_brightness = brightness
            brightest_image = name

    return brightest_image

def compare_edges(images):
    """
    This function returns the name of the sharpest image.
    :param images: Dict of image names and their respective image data.
    :return: Name of the image with the most edges.
    """
    max_edeges = 0
    image_with_most_edges = None

    for name,img in images.items():
        edges = cv2.Canny(img, 100, 200)
        edges_count = np.sum(edges > 0)

        if edges_count > max_edeges:
            max_edeges = edges_count
            image_with_most_edges = name


    return image_with_most_edges
