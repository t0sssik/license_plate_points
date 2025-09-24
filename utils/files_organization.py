import os
import shutil
supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

"""
Тут переношу изображения и аннотации из кучи папок в 1 папку для каждого типа номера
"""

root_dir = "data/lpr_keypoints"
dest_rect_images = "data/rect/images"
dest_rect_annot = "data/rect/labels"
dest_square_images = "data/square/images"
dest_square_annot = "data/square/labels"

os.makedirs(dest_rect_images, exist_ok=True)
os.makedirs(dest_rect_annot, exist_ok=True)
os.makedirs(dest_square_images, exist_ok=True)
os.makedirs(dest_square_annot, exist_ok=True)

rect_image_count = 0
rect_annot_count = 0
square_image_count = 0
square_annot_count = 0
duplicate_files = set()

for root, dirs, files in os.walk(root_dir):
    if "RECT" in root.upper():
        dest_images_dir = dest_rect_images
        dest_annot_dir = dest_rect_annot
        dataset_type = "RECT"
    elif "SQUARE" in root.upper():
        dest_images_dir = dest_square_images
        dest_annot_dir = dest_square_annot
        dataset_type = "SQUARE"
    else:
        continue
    
    for file in files:
        file_path = os.path.join(root, file)
        if file.lower().endswith(supported_formats):
            dest_image_path = os.path.join(dest_images_dir, file)
            if os.path.exists(dest_image_path):
                duplicate_files.add(file)
                name, ext = os.path.splitext(file)
                counter = 1
                while os.path.exists(os.path.join(dest_images_dir, f"{name}_{counter}{ext}")):
                    counter += 1
                dest_image_path = os.path.join(dest_images_dir, f"{name}_{counter}{ext}")
            shutil.copy2(file_path, dest_image_path)
            if dataset_type == "RECT":
                rect_image_count += 1
            else:
                square_image_count += 1
        
        elif file.endswith('.txt'):
            dest_annot_path = os.path.join(dest_annot_dir, file)
            if os.path.exists(dest_annot_path):
                duplicate_files.add(file)
                name, ext = os.path.splitext(file)
                counter = 1
                while os.path.exists(os.path.join(dest_annot_dir, f"{name}_{counter}{ext}")):
                    counter += 1
                dest_annot_path = os.path.join(dest_annot_dir, f"{name}_{counter}{ext}")            
            shutil.copy2(file_path, dest_annot_path)
            if dataset_type == "RECT":
                rect_annot_count += 1
            else:
                square_annot_count += 1