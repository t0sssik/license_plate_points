import argparse
import cv2
import numpy as np
import onnxruntime as ort

def preprocess(img, stride=32):
    h, w = img.shape[:2]
    new_h = int(np.ceil(h / stride) * stride)
    new_w = int(np.ceil(w / stride) * stride)
    img_resized = cv2.resize(img, (new_w, new_h))
    return img_resized

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Путь до onnx модели")
parser.add_argument("--input_path", type=str, required=True, help="Путь до входного изображения")
parser.add_argument("--output_path", type=str, required=True, help="Путь до выходного изображения")
args = parser.parse_args()

session = ort.InferenceSession(args.model)
input_name = session.get_inputs()[0].name

img = cv2.imread(args.input_path)
orig_h, orig_w = img.shape[:2]

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = preprocess(img_rgb)
new_h, new_w = img_resized.shape[:2]

img_input = img_resized.astype(np.float32) / 255.0
img_input = np.transpose(img_input, (2, 0, 1))
img_input = np.expand_dims(img_input, axis=0)

outputs = session.run(None, {input_name: img_input})
pred = outputs[0][0]

scale_x = orig_w / new_w
scale_y = orig_h / new_h

for det in pred:
    x, y, w, h, conf = det[:5]
    if conf < 0.5:
        continue

    kpts = det[6:]
    for i in range(0, len(kpts), 2):
        kx, ky = kpts[i:i+2]
        kx = int(kx * scale_x)
        ky = int(ky * scale_y)
        cv2.circle(img, (kx, ky), 2, (0, 0, 255), -1)

cv2.imwrite(args.output_path, img)
print(f"Результат сохранен в {args.output_path}")