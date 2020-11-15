import cv2
import pickle
from imutils.video import WebcamVideoStream
import numpy as np
from tensorflow import keras
import tensorflow as tf


class VideoCamera(object):
    def __init__(self):
        self.stream = WebcamVideoStream(src=0).start()
        self.model=keras.models.load_model('my_model.h5')

    def __del__(self):
        self.stream.stop()
    
    def get_frame(self):
        self.model=keras.models.load_model('my_model.h5')

        im = self.stream.read()
        labels_dict={0:'without_mask',1:'with_mask'}
        color_dict={0:(0,0,255),1:(0,255,0)}
        size = 4
        classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        im=cv2.flip(im,1,1) #Flip to act as a mirror

        # Resize the image to speed up detection
        mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

        # detect MultiScale / faces 
        faces = classifier.detectMultiScale(mini)

        # Draw rectangles around each face
        for f in faces:
            (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
            #Save just the rectangle faces in SubRecFaces
            face_img = im[y:y+h, x:x+w]
            resized=cv2.resize(face_img,(150,150))
            normalized=resized/255.0
            reshaped=np.reshape(normalized,(1,150,150,3))
            reshaped = np.vstack([reshaped])
            
            result=self.model.predict(reshaped)
            #print(result)
            
            label=np.argmax(result,axis=1)[0]
        
            cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
            cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
            cv2.putText(im, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)


        ret, jpeg = cv2.imencode('.jpg', im)
        data = []
        data.append(jpeg.tobytes())
        return data

