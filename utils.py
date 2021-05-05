import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os
import pandas as pd

#####################################################################
# DATA PREPARATION

def create_dataset():
    loc = os.getcwd()
    path = loc + '/myData'
    data = []
    labels = []

    for i in range(10):
        files = os.listdir(path +"/"+str(i))
        for file in files:
            path_file = path + "/" + str(i) + "/" + file
            data.append(path_file)
            labels.append(i)

    dictP_n = {"X": data,
              "Y": labels}   

    data  = pd.DataFrame(dictP_n, index = None)
    data = data.sample(frac = 1)
    data.to_csv("test.csv", index =None)

def make_it_better(path):
    x=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x,(64,64))
    x = (255-x)
    x = x.astype('float32')
    x /= 255
    x = np.asarray(x)
    return x

def make_it_better_d(d):
    x = cv2.resize(d,(64,64))
    x = (255-x)
    x = x.astype('float32')
    x /= 255
    x = np.asarray(x)
    xd = x.reshape((-1, 64, 64, 1))
    return xd, x

def load_dataset(path):
    df = pd.read_csv(path)
    return df
#####################################################################################################
# IMAGE PREPARATION

def preprocessing(img):
    imgbw = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    proc = cv2.GaussianBlur(imgbw, (3, 3), 1)

    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C
                                , cv2.THRESH_BINARY, 15, 6)
    proc = cv2.bitwise_not(proc, proc)

    return proc

def getContours(img, img2):
    biggest = np.array([])
    maxArea = 0
    contours,hierarhy = cv2.findContours(img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for el in contours:
        area = cv2.contourArea(el)
        if area > 5000:
            cv2.drawContours(img2, el, -1, (0,255,0), 3)
            par = cv2.arcLength(el,True)
            approx = cv2.approxPolyDP(el,0.02*par, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    #cv2.drawContours(img2, biggest, -1, (0,255,0), 10)
    return biggest

def smol_cnt(img):
    thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                   cv2.THRESH_BINARY_INV,39,10)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vertical_kernel, iterations=2)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horizontal_kernel, iterations=2)

    return thresh


def get_mini_squares(img, img2):
    #img = smol_cnt(img)
    contours, h = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    for el in contours:
        if cv2.contourArea(el) > 400:
            x, y, w, h = cv2.boundingRect(el)
            cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img


def reorganize(points):
    rp = points.reshape((4,2))
    rpn = np.zeros((4,1,2), np.int32)
    add  = rp.sum(1)

    rpn[0] = rp[np.argmin(add)]
    rpn[3] = rp[np.argmax(add)]
    
    diff = np.diff(rp,axis=1)

    rpn[1] = rp[np.argmin(diff)]
    rpn[2] = rp[np.argmax(diff)]

    return rpn

def getWarp(img, biggest):
    biggest = reorganize(biggest)
    p1 = np.float32(biggest)
    p2 = np.float32([[0,0],[img.shape[1],0], [0,img.shape[0]],
                     [img.shape[1], img.shape[0]]])
    matrix = cv2.getPerspectiveTransform(p1,p2)
    imgWarped = cv2.warpPerspective(img,matrix,(img.shape[1],img.shape[0]))

    return imgWarped

def get_digit(img):
    w = img.shape[0]
    h = img.shape[1]
    sqw = w // 9
    sqh = h // 9
    digits = []
    for i in range(9):
        for j in range(9):
            a = img[i*sqw:(i+1)*sqw, j*sqh:(j+1)*sqh]
            a = cv2.resize(a,(64,64))
            a = (255-a)
            a = np.asarray(a)
            digits.append(a)

    return digits

###################################################################
# Sudoku solver
def sudoku(p):
    if solve(p):
        return p
    

def solve(puzzle):
    f = find_empty(puzzle)
    if not f:
        return True
    else:
        p1, p2 = f
    
    for i in range(1,10):
        if check(puzzle, i, p1, p2):
            puzzle[p1][p2] = i
            if solve(puzzle):
                return True
            puzzle[p1][p2] = 0
            
                    
    return False

def find_empty(p):
    for i, row in enumerate(p):
        for j, el in enumerate(row):
            if el  == 0:
                return i, j
    return None

def check(pu, n, p1, p2):
    for i in range(len(pu)):
        if pu[p1][i] == n and p2 != i:
            return False
    for i in range(len(pu)):
        if pu[i][p2] == n and p1 != i:
            return False
    
    b_x = p2 // 3
    b_y = p1 // 3
    
    for i in range(b_y*3, b_y*3+3):
        for j in range(b_x * 3, b_x*3 + 3):
            if pu[i][j] == n and (i,j) != (p1,p2):
                return False

    return True
        
