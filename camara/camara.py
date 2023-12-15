import cv2
import numpy as np
import tensorflow as tf 
from keras.models import load_model


#expresion_name = ["Enfadado", "Feliz", "Triste", "Sorprendido", "Neutral"]
expresion_name = ["Enfadado", "Feliz", "Sorprendido", "Triste"]
model = load_model(filepath='./modelo/reconocimientoFacial.h5', compile=False)
#model = load_model(filepath='./modelo/prueba/prueba2.h5', compile=False)






cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        #frame = cv2.flip(frame,1) 
        cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = cascade.detectMultiScale(gray, 1.1, 4)
        a,b,c,d = 0,0,10,10
        for (x, y, w, h) in face:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            a,b,c,d = x,y,w,h
            if len(face) > 0:
                top, bottom, left, right = b, b+d, a, a+c
                image = frame[top:bottom, left:right]
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image,(48,48), interpolation=cv2.INTER_AREA)
                image = image/255
                #image = np.expand_dims(image, axis=-1)
                image = np.expand_dims(image, axis=0)
                emotion = model(image, training=False)
                num = np.argmax(emotion, axis=1)[0]
                emotion = expresion_name[num]
                #cv2.putText(frame, emotion, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)
                cv2.putText(frame, emotion, (face[0][0], face[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)
        cv2.imshow('Detector de emociones', frame)
        if cv2.waitKey(1) == 27:
            break
    else:
        raise RuntimeError('Error al leer de la cámara.')

cap.release()
cv2.destroyAllWindows()
