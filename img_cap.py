from utils import *
from modeltf import *
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from os import path



if __name__ == '__main__':
    if not path.exists('test.csv'):
        create_dataset()

    if path.exists("my_model"):
        model = load_model('my_model')
    else:
        model = create_model()
        model = learn_model(model)

    cap = cv2.VideoCapture(0)
    cap.set(3,720)
    cap.set(4,720)
    cap.set(10,150)
    board = np.zeros(81)
    itr = 0
    

    while True:
        success, img = cap.read()
        imgContour  = img.copy()
        imgThres = preprocessing(img)
        biggest = getContours(imgThres, imgContour)

        if biggest.size != 0:
            

            #imgTT = getWarp(img, biggest)
            imgW = getWarp(imgThres, biggest) 
            digits = get_digit(imgW)

            #
            #imgS = smol_cnt(imgW)
            #imgT = get_mini_squares(imgS, imgTT)
            #cv2.imshow('img', imgS)
            #cv2.imshow('imgW', imgW)
            #cv2.imshow('imgT', imgT)
            #cv2.imshow('tr', imgTT)
            if itr == 0:
                y = np.zeros((81,10,1))
            digits = get_digit(imgW)
            #final_sudo = []
            #prt = True
            #if itr < 81*4:
            #    print(itr)
            #    for i in range(len(digits)):
            #        du = cv2.resize(digits[i], (64,64))
            #        d = model.predict(du.reshape(1,64,64,1))
            #        d = d.argmax(axis=-1)
            #        #print(d)
            #        y[i][d] =+ 1
            #        itr += 1
            #elif prt == True:
            #    dgts = []
            #    print(y[:30].argmax(axis=-1))
            #    prt = False
            xD, Dx = make_it_better_d(digits[2])
            print(model.predict(xD))
            cv2.imshow('1',Dx)
            
        cv2.imshow('poka', imgContour)
        if cv2.waitKey(1)& 0xFF == ord('q'):
            break
