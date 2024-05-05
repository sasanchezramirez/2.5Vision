import cv2
import numpy as np

def read_image(img_1, img_2):
    img_1 = cv2.imread(img_1)
    img_2 = cv2.imread(img_2)
    img_1_processed = cv2.cvtColor(np.array(img_1), cv2.COLOR_RGB2BGR)
    img_2_processed = cv2.cvtColor(np.array(img_2), cv2.COLOR_RGB2BGR)
    message = compare_image(img_1_processed, img_2_processed)
    print (f"Retorna {message}")
    return message

def compare_image(img_1, img_2):
    results = {}
    results['brightness_comparison'] = compare_brightness(img_1, img_2)
    results['edge_visibility_comparison'] = compare_edges(img_1, img_2)
    return results

def compare_brightness(img_1, img_2):
    hsv_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2HSV)
    hsv_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2HSV)
    mean_val_1 = cv2.mean(hsv_1)[2]
    mean_val_2 = cv2.mean(hsv_2)[2]
    return {"Image 1 Brightness": mean_val_1, "Image 2 Brightness": mean_val_2}

def compare_edges(img_1, img_2):
    edges_1 = cv2.Canny(img_1, 100, 200)
    edges_2 = cv2.Canny(img_2, 100, 200)
    return {"Image 1 Edge Count": np.sum(edges_1 > 0), "Image 2 Edge Count": np.sum(edges_2 > 0)}
