import tflite_runtime.interpreter as tflite
import numpy as np
import requests as req
import re
import time
import datetime
from picamera import PiCamera
from PIL import Image, ImageOps
from io import BytesIO

target = 16

def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor

def load_labels(path):
    """Loads the labels file. Supports files with or without index numbers."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = {}
        for row_number, content in enumerate(lines):
            pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number] = pair[0].strip()
        return labels

def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    # Get all output details
    # boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

    for i in range(count):
        # Cat is class #16
        if classes[i] == target and scores[i] >= threshold:
            print(scores[i])
            return True
    return False

# Webserver to send requests to
url = 'http://192.168.50.157:5000/catcam'

# Load the TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path="ssd_mobilenet_v1.tflite")
interpreter.allocate_tensors()

# Load the output labels
labels = load_labels("coco_labels.txt")

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with PiCamera(resolution=(300,300)) as camera:

    time.sleep(1)

    # Create input stream
    stream = BytesIO()
    for foo in camera.capture_continuous(stream, format="jpeg"):

        try:
 
            # Load image
            img = Image.open(stream).convert('RGB')
            captureTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            #img.save(captureTime + ".jpg")
    
            # Detect cat
            cat_present = detect_objects(interpreter, img, 0.15)
            print("cat present:" , cat_present)
        
            # Send POST request to webserver
            payload = {'status': int(cat_present)}
            myHeaders = {'Content-type': 'application/json'}
            r = req.post(url, json = payload, headers = myHeaders)
            if r.status_code == 200:
                content = r.json()
                print(captureTime,content)
                # Cooldown timer if dispenser is activated
                if content['action'] == 'dispensed':
                    print("Waiting 1 min")
                    time.sleep(60)
                else:
                    print("wait 0.5 sec")
                    time.sleep(0.5)
            else:
                print("server error: " , r)

        except Exception as e:
            print("exception:",e.args)

        finally: 
            # Clear stream and reset position
            stream.truncate(0)
            stream.seek(0)
        

