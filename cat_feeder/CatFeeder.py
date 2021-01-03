import tflite_runtime.interpreter as tflite
import flask
from flask import request
from picamera import PiCamera
from PIL import Image, ImageOps
import numpy as np
from io import BytesIO
import foodpi

app = flask.Flask(__name__)
#app.config["DEBUG"] = True

# Load the TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path="/home/pi/bowl-ml/cat_feeder/bowl-4998.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# State of each camera
catStatus = 0
bowlStatus = 0

# Initialize bowl camera
bowlCamera = PiCamera(resolution=(300,300))

# Return 1 if bowl is not empty
def detect_bowl():
    
    # Create input stream and capture image
    stream = BytesIO()
    bowlCamera.capture(stream, format='jpeg')
    
    # Reset position
    stream.seek(0)
    
    # Load image into np array
    img = Image.open(stream)
    img = ImageOps.grayscale(img)
    img_arr = np.array(img)
    img_arr = img_arr.astype(np.float32)
    img_arr = np.expand_dims(img_arr,[0,3])
    img_arr /= 255.0

    # Test the model
    interpreter.set_tensor(input_details[0]['index'], img_arr)
    interpreter.invoke()

    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data[0][0] <= 0.5
    print(output_data, prediction)
    return prediction

# Home page, displays debug info
@app.route('/', methods=['GET'])
def home():
    return "<h1>Cat food dispenser API is running</h1><br />"

# Cam URL, handle POST requests from cameras
@app.route('/catcam', methods=['POST'])
def cam_handler():

    # Get status from cat detector
    content = request.get_json()
    catStatus = content['status']

    # Get status from bowl detector
    if catStatus == 1:
        bowlStatus = detect_bowl()
        if bowlStatus == 1:
            print("DISPENSING")
            foodpi.dispense()
            return '{"action":"dispensed"}'
    return '{"action":"nothing"}'

# Shutdown server
def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

@app.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

if( __name__ == '__main__'):
    app.run(host="0.0.0.0", port=5000)

