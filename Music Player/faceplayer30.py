import imutils
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import face_recognition as fc
import vlc
import time

model=load_model('_mini_XCEPTION.106-0.65.hdf5')
fd=cv2.CascadeClassifier(r'C:\Program Files\Python36\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml')
em=['grumpy','disgust','fear','happy','sad','surprised','neutral']
v=cv2.VideoCapture(0)
exp=''
act='play'
path1='E:\PYTHON\Programs class\Music Player\prog\\'
path=path1+'happy'+'.mp3'
pl=vlc.MediaPlayer(path)

while(1):
    r,i=v.read()
    gray=cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
    faces=fd.detectMultiScale(gray)
    for [x,y,w,h] in faces:
            roi=gray[y:y+h,x:x+w]
            roi=cv2.resize(roi,(48,48))
            roi=roi.astype('float')/255.0
            roi=img_to_array(roi)
            roi=np.expand_dims(roi,axis=0)
            p=list(model.predict(roi)[0])
            exp=em[p.index(max(p))]
            #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            #print(em[p.index(max(p))])
            #print(p)

    cv2.imshow('frame',i)
    k=cv2.waitKey(5)
    if(k==ord('q')):
        v.release()
        pl.stop()
        cv2.destroyAllWindows()
        
    if(k==ord('m')):
        if(act=='play'):
            pl.stop()
            path1='E:\PYTHON\Programs class\Music Player\prog\\'
            path=path1+exp+'.mp3'
            pl=vlc.MediaPlayer(path)
            
            print(exp)
            pl.play()

    if(k==ord('p')):
        pl.pause()
        




