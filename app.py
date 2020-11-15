from flask import Flask, render_template, request, Response
from camera import VideoCamera
import cv2
import numpy as np
from tensorflow import keras
model=keras.models.load_model('my_model.h5')
  
app = Flask(__name__)
@app.route('/')
def home():
  return render_template('index.html')
def gen(camera):
    while True:
        data= camera.get_frame()
        frame=data[0]
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/result',methods=['POST'])
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')



  
  
  
if __name__=='__main__':
  app.run(debug=True)