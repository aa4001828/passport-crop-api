from flask import Flask, request, send_file
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

def crop_passport_size_from_bytes(image_bytes):
    npimg = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(rgb, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    margin_y = int(h * 1.8)
    margin_x = int(w * 1.5)
    cx = x + w // 2
    cy = y + h // 2

    top = max(cy - margin_y // 2, 0)
    bottom = min(cy + margin_y // 2, image.shape[0])
    left = max(cx - margin_x // 2, 0)
    right = min(cx + margin_x // 2, image.shape[1])

    cropped = image[top:bottom, left:right]
    final = cv2.resize(cropped, (600, 800))

    img_pil = Image.fromarray(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    img_io = io.BytesIO()
    img_pil.save(img_io, format='JPEG')
    img_io.seek(0)
    return img_io

@app.route('/crop', methods=['POST'])
def crop_image():
    if 'image' not in request.files:
        return {"error": "No file provided"}, 400

    image = request.files['image']
    result = crop_passport_size_from_bytes(image.read())

    if result is None:
        return {"error": "No face detected"}, 400

    return send_file(result, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run()
