from flask import Flask, request
from predictSingleDehazed import predict, pretrain_load
from PIL import Image
import numpy as np
import cv2 as cv
import base64
from flask_cors import CORS


app = Flask(__name__)
cors = CORS(app)
gen, config, args = pretrain_load()


@app.route('/predict', methods=['POST'])
def handle():
    data = request.files.get('image')
    image = Image.open(data)
    img = np.array(image, dtype=np.float32)
    result = predict(config, args, gen, img)
    _, im_arr = cv.imencode('.jpg', result)
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    return im_b64