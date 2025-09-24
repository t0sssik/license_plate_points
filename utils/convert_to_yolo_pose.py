import os

def convert_to_yolo_pose(txt_path):
    """
    Конвертирует разметку TL/TR/BR/BL в формат YOLO-Pose:
    class_id center_x center_y width height kp1_x kp1_y kp2_x kp2_y kp3_x kp3_y kp4_x kp4_y
    """
    points = {}
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            key, x, y = parts
            points[key] = (float(x), float(y))
    
    x1, y1 = points['TL']
    x2, y2 = points['TR']
    x3, y3 = points['BR']
    x4, y4 = points['BL']
    
    all_x = [x1, x2, x3, x4]
    all_y = [y1, y2, y3, y4]
    
    x_min = min(all_x)
    x_max = max(all_x)
    y_min = min(all_y)
    y_max = max(all_y)
    
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    yolo_line = f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}\n"
    
    with open(txt_path, 'w') as f:
        f.write(yolo_line)
                

root_dir = "data/lpr_keypoints/"
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".txt"):
            txt_path = os.path.join(root, file)
            convert_to_yolo_pose(txt_path)