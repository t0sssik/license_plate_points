import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def draw_yolo_pose_points(image, annotation_path, point_color=(255, 0, 0), point_size=2):
    if isinstance(image, str):
        img = cv2.imread(image)
    elif isinstance(image, Image.Image):
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = image.copy()
    
    height, width = img.shape[:2]
    
    with open(annotation_path, 'r', encoding='utf-8') as file:
        annotation = file.readline()
    parts = annotation.strip().split()
    coords = [float(x) for x in parts[5:]] # без класса и bbox
    for i in range(0, len(coords), 2):
        x = int(coords[i] * width)
        y = int(coords[i + 1] * height)
        cv2.circle(img, (x, y), point_size, point_color, -1)
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()
    return img_rgb