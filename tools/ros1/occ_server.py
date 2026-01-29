#!/usr/bin/env python
import io
import json
import time

from flask import Flask, jsonify, request
from PIL import Image

app = Flask(__name__)


def load_image(file_storage):
    if file_storage is None:
        return None
    image_bytes = io.BytesIO(file_storage.read())
    return Image.open(image_bytes).convert('RGB')


def run_inference(images, metadata):
    time.sleep(0.01)
    widths = [img.size[0] for img in images if img is not None]
    heights = [img.size[1] for img in images if img is not None]
    return {
        'status': 'ok',
        'num_images': len([img for img in images if img is not None]),
        'avg_width': sum(widths) / len(widths) if widths else 0,
        'avg_height': sum(heights) / len(heights) if heights else 0,
        'metadata': metadata,
    }


@app.route('/infer', methods=['POST'])
def infer():
    start = time.time()
    metadata = {}
    if 'json' in request.form:
        try:
            metadata = json.loads(request.form['json'])
        except json.JSONDecodeError:
            metadata = {'raw': request.form['json']}

    img_left = load_image(request.files.get('image_left'))
    img_right = load_image(request.files.get('image_right'))
    img_front = load_image(request.files.get('image_front'))

    result = run_inference([img_left, img_right, img_front], metadata)
    result['elapsed_sec'] = time.time() - start
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5801)
