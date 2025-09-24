import argparse
import cv2
import time
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt

def letterbox(img, input_size=(416, 416)):
    h, w = img.shape[:2]
    new_w, new_h = input_size
    scale = min(new_w / w, new_h / h)
    resized_w, resized_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (resized_w, resized_h))
    dw = new_w - resized_w
    dh = new_h - resized_h
    top, bottom = dh//2, dh - dh//2
    left, right = dw//2, dw - dw//2
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(114,114,114))
    return img_padded

def save_and_show_model_pred(img, pred, path, point_color=(255,0,0), point_size=3):
    img_np = img.squeeze(0).squeeze(0)
    img_np = cv2.cvtColor((img_np*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    h, w = img_np.shape[:2]
    for i in range(0, len(pred), 2):
        x = int(pred[i] * w)
        y = int(pred[i+1] * h)
        cv2.circle(img_np, (x, y), point_size, point_color, -1)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, img_rgb)
    print(f"Результат сохранен в {path}")
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb)
    plt.title('Результат работы модели')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Путь до onnx модели")
parser.add_argument("--input_path", type=str, required=True, help="Путь до входного изображения")
parser.add_argument("--output_path", type=str, required=True, help="Путь до выходного изображения")
args = parser.parse_args()

session = ort.InferenceSession(args.model)
input_name = session.get_inputs()[0].name

img = cv2.imread(args.input_path, cv2.IMREAD_GRAYSCALE)
img_input = letterbox(img)
img_input = img_input.astype(np.float32) / 255.0
img_input = np.expand_dims(img_input, axis=0)
img_input = np.expand_dims(img_input, axis=0)

start = time.time()
outputs = session.run(None, {input_name: img_input})
end = time.time()
print(f"Время инференса модели: {(end - start)*1000:.2f} мс")

pred = outputs[0][0]

save_and_show_model_pred(img_input, pred, args.output_path)