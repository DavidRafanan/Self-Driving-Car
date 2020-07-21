#RUN DOING PYTHON3.6
#conda create --name stupid_conda_shit
#conda activate stupid_conda_shit
#conda install python=3.6    
#conda install -c anaconda flask
#conda install -c conda-forge python-socketio
#conda install -c conda-forge eventlet
#conda install -c conda-forge tensorflow
#conda install -c conda-forge keras
#conda install -c anaconda pillow  
#conda install -c anaconda numpy
#conda install -c conda-forge opencv

import socketio #realtime communication between client and server
import eventlet
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

sio = socketio.Server() #connects to Server

app = Flask(__name__) #instance of Flask class (__name__ = __main__)
speed_limit = 10

#from behavioral cloning code
def img_preprocess(img): 
  #img = mpimg.imread(img)
  img = img[60:135,:,:] #3d array - height, width, channel ---height is obtaining the road 

  #COMPUTER VISION
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV) #YUV effective for training in NVIDIA model
  img = cv2.GaussianBlur(img, (3,3), 0) #gaussian blur smoothens image --- less noise
  img = cv2.resize(img, (200,66))

  #REQUIRED FOR PREPROCESSING
  img = img/255
  return img

@sio.on('telemetry') #event handler for telemetry function
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image']))) #image is base64 encoded
    image = np.asarray(image) #converts image to array
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed/speed_limit #ensures constant speed at speed limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)

#what url to use to trigger greeting function
#@app.route('/home')
#def greeting():
#    return 'Welcome!'

#when connection to client, fire event handler
@sio.on('connect') #event handler for connect function --- also sends message, disconnect
def connect(sid, environ):
    print('Connected')
    send_control(0, 0) #drives straight (0) no throttle (0) throttle (1)

def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })

if __name__ == '__main__':
    #app.run(port=3000) #listens at localhost:3000
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app) #connects server to app 
    eventlet.wsgi.server(eventlet.listen(('',4567)), app) #sends request from any IP ('') to app through port 4567