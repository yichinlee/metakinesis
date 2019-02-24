from __future__ import print_function
import numpy as np
import cv2
from PIL import Image
from PIL import ImageTk
import ffmpeg
import math
import random
import noise
#from pypi import ghos
from tkinter import *
from os.path import realpath, normpath
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from PIL import ImageGrab
from numpy import array
import os
import PIL.Image

def rgbColor (rgb):
    return "#%02x%02x%02x" % rgb   

class circle(object):
    def __init__(self,center,color):
        self.cx = center[0] 
        self.cy = center[1] 
        self.color = color
        self.cirDim = 35

    def move(self,data,x,y):
        dx = x * 0.01
        dy = y * 0.01
    
        #check the top-left point
        #if out of boundary, move neg-direction
        #cx
        if data.cirGrid[0][0][0]<0 and data.moveRight == True:
            self.cx += dx

        elif data.cirGrid[0][0][0]> -data.width/2 and data.moveLeft == True:
            self.cx -= dx

        elif data.cirGrid[0][0][0] >= 0:
            data.moveLeft = True
            data.moveRight = False

        elif data.cirGrid[0][0][0] <= -data.width/2:
            data.moveLeft = False
            data.moveRight = True

        #cy
        if data.cirGrid[0][0][1]<0 and data.moveDown == True:
            self.cy += dy

        elif data.cirGrid[0][0][1]> -data.width/2 and data.moveUp == True:
            self.cy -= dx

        elif data.cirGrid[0][0][1] >= 0:
            data.moveUp = True
            data.moveDown = False

        elif ddata.cirGrid[0][0][1] <= -data.width/2:
            data.moveUp = False
            data.moveDown = True

    def scale(self,data,x,y):
        #radious depands on the 
        dist = abs((self.cx - x)**2 - (self.cy - y)**2)
        scaleVal = math.sqrt(dist) * 0.01
        #self.cirDim -= scaleVal

        if self.cirDim < 45 and data.radInc == True:
            self.cirDim += scaleVal
        
        elif self.cirDim > 0 and data.radDec == True:
            self.cirDim -= scaleVal

        elif self.cirDim <= 0:
            data.radInc = True
            data.radDec = False

        elif self.cirDim >= 35:
            data.radInc = False
            data.radDec = True

    def translate(self, value, leftMin, leftMax, rightMin, rightMax):
        
        leftSpan = leftMax - leftMin
        rightSpan = rightMax - rightMin
        # Convert the left range into a 0-1 range (float)
        valueScaled = float(value - leftMin) / float(leftSpan)
        # Convert the 0-1 range into a value in the right range
        return rightMin + (valueScaled * rightSpan)

    def changeColor(self,x,y,RGB):
        if RGB == 1:
            dist = abs((self.cx - x)**2 - (self.cy - y)**2)
            colorVal = math.sqrt(dist) * 0.1
            RVal = self.translate(colorVal, 0, 200, 0,255)
            RVal = int(RVal)
            self.color = '#%02x%02x%02x' % (RVal, 255 -RVal , int(RVal*0.5))
            #print("R:",RVal, 255 -RVal , int(RVal*0.5))

        elif RGB == 2:
            dist = abs((self.cx - x)**2 - (self.cy - y)**2)
            colorVal = math.sqrt(dist) * 0.1
            GVal = self.translate(colorVal, 0, 200, 0,255)
            GVal = int(GVal)
            self.color = '#%02x%02x%02x' % (int(255 - GVal * 0.2) , GVal, int(GVal * 0.2))
            #print("G:",int(255 - GVal * 0.8) , GVal, int(GVal * 0.8))

        elif RGB == 3:
            dist = abs((self.cx - x)**2 - (self.cy - y)**2)
            colorVal = math.sqrt(dist) * 0.1
            BVal = self.translate(colorVal, 0, 200, 0,255)
            BVal = int(BVal)
            self.color = '#%02x%02x%02x' % (0, int(BVal*0.5) , BVal)
            #print("B=",0, BVal , BVal)

    def drawCircle(self, canvas):
        canvas.create_oval (self.cx, self.cy, self.cx + self.cirDim, self.cy + self.cirDim, outline = self.color, fill = self.color)

#rect in mode 2
class rotatingSquare(object):
    def __init__(self, center, angle, noise):
        self.angle = angle
        self.startingAng = 90
        self.noise = noise
        self.squareRadius = 20
        self.cx = center[0]
        self.cy = center[1]
        self.cornerPoints = []
        self.counter = 0

        #original position
        for i in range(4):
            rotate = math.radians(self.startingAng+(90*i))
            x = self.cx + self.squareRadius * math.cos(rotate)
            y = self.cy + self.squareRadius * math.sin(rotate)
            self.cornerPoints.append(x)
            self.cornerPoints.append(y)


    def rotate(self,data, angle):
        #change the cornerPoints, update the drawing
        self.counter +=1
        wave = math.radians(self.cx)
        waveVal = math.sin(wave)
        self.startingAng += angle * waveVal * (self.cy) * 0.01
        for i in range(0,8,2):
            rotVal = math.radians(self.startingAng+(90*(i//2))) 
            self.cornerPoints[i] = self.cx + self.squareRadius * math.cos(rotVal)
            self.cornerPoints[i+1] = self.cy + self.squareRadius * math.sin(rotVal)

    def scale(self,data,noise):
        #input noise position of lower
        dist = abs((self.cx - noise[0])**2 - (self.cy - noise[1])**2)
        scaleFactor = math.sqrt(dist) *0.01
        if self.squareRadius < 35 and data.radInc == True:
            self.squareRadius += scaleFactor
        elif self.squareRadius > 0 and data.radDec == True:
            self.squareRadius -= scaleFactor

        elif self.squareRadius <= 0:
            data.radInc = True
            data.radDec = False

        elif self.squareRadius >= 35:
            data.radInc = False
            data.radDec = True

    def drawSquare(self, canvas):
        canvas.create_polygon(self.cornerPoints,fill ="black")

#rect in mode 3
class rotatingRect(object):
    def __init__(self,anchor,angle):
        #(ax,ay)anchor points
        self.ax = anchor[0]
        self.ay = anchor[1]
        self.angle = angle
        self.rectPoints = []
        self.noise = 0
        self.rectColor = "black"
        
        # short = 20 long = 60 diagonal = sqrt(20**2 + 60**2)
        r0 = 0
        r1 = 12
        r2 = 60
        r3 = math.sqrt(r1**2 + r2**2)
        self.radiusSet = [r1,r3,r2,r0]
        self.rectTheta = math.atan(5)
        self.angleSet = [0, self.rectTheta, 3.14/2, 0]
        self.rectPoints = [0] * 8

        #original position
        for i in range (0,4):
            rad = self.angleSet[i]
            cosVal = math.cos(rad)
            sinVal = math.sin(rad)
            self.rectPoints[2*i] = self.ax + self.radiusSet[i] * cosVal
            self.rectPoints[(2*i)+1] = self.ay + self.radiusSet[i] * sinVal

    def rotate(self,data,factor):
        temp = abs((factor[0] - self.ax)**2 - (factor[1] - self.ay)**2)
        dist = math.sqrt(temp)
        self.angle = dist * 0.1

        for i in range (0,4):
            rotRad = math.radians(self.angle)
            rad = self.angleSet[i] + rotRad
            cosVal = math.cos(rad)
            sinVal = math.sin(rad)
            self.rectPoints[2*i] = self.ax + self.radiusSet[i] * cosVal
            self.rectPoints[(2*i)+1] = self.ay + self.radiusSet[i] * sinVal

    def translate(self, value, leftMin, leftMax, rightMin, rightMax):
        
        leftSpan = leftMax - leftMin
        rightSpan = rightMax - rightMin
        # Convert the left range into a 0-1 range (float)
        valueScaled = float(value - leftMin) / float(leftSpan)
        # Convert the 0-1 range into a value in the right range
        return rightMin + (valueScaled * rightSpan)

    def changeColor(self,data):
        #colorArea are a area - 2D list
        j = self.translate(self.ax,0,1500,0,1280)
        i = self.translate(self.ay,0,1500,0,720)
        # i = int(self.ax) 
        # j = int(self.ay) 
        i = int(i)
        j = int(j)
        #print("i,j =",i,j)
        #print(data.pxVal[i][j][0])
        if data.pxVal[i][j][0] > 100:
        	self.rectColor = "cyan"
        
    def drawRect(self,canvas):

        canvas.create_polygon(self.rectPoints,fill = self.rectColor)

#core animation code
def init(data):
    #there are 3 mode 
    data.mode = 3
    # load data.xyz as appropriate
    #change the video name here
    data.cap = cv2.VideoCapture('testing_5.mp4')
    data.face_cascade = cv2.CascadeClassifier(normpath(realpath(cv2.__file__) + '../../../../../share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'))
    data.low_cascade = cv2.CascadeClassifier(normpath(realpath(cv2.__file__) + '../../../../../share/OpenCV/haarcascades/haarcascade_lowerbody.xml'))
    data.upper_cascade = cv2.CascadeClassifier('opencv-master/data/haarcascades/haarcascade_upperbody.xml')
    data.eye_cascade = cv2.CascadeClassifier('opencv-master/data/haarcascades/haarcascade_eye.xml')

    data.frame =[]
    data.timer = 0
    data.width = 1280
    data.height = 720

    #data for circle gird
    data.moveUp = False
    data.moveDown = True
    data.moveLeft = False
    data.moveRight = True

    data.cirGridLeft = - data.width/2
    data.cirGridTop = - data.height/2
    data.cirGridWidth = data.width * 2
    data.cirGridHeight = data.height * 2
    data.deltaLowerX = 0
    data.deltaLowerY = 0
    data.circleElem_R = []
    data.circleElem_G = []
    data.circleElem_B = []

    data.circleColor_R = []
    data.circleColor_G = []
    data.circleColor_B = []
    #initial circle grid
    #circle diameter = 35, gird = 50
    data.cirGridSize = 50
    data.cirRows = data.cirGridWidth // 50
    data.cirCols = data.cirGridHeight //50
    data.cirGrid = [([0] * data.cirCols) for row in range(data.cirRows)]

    #layer 1:R control by lower body
    for row in range(len(data.cirGrid)):
        for col in range(len(data.cirGrid[0])):
            data.cirGrid[row][col] = (data.cirGridLeft + data.cirGridSize * row, 
                data.cirGridTop + data.cirGridSize * col) 
            if  data.circleColor_R == []:
                data.circleColor_R = "magenta2"
            data.circleElem_R.append(circle(data.cirGrid[row][col],data.circleColor_R))

    #layer 2:G control by upper body
    for row in range(len(data.cirGrid)):
        for col in range(len(data.cirGrid[0])):
            data.cirGrid[row][col] = (data.cirGridLeft + data.cirGridSize * row, 
                data.cirGridTop + data.cirGridSize * col) 
            if  data.circleColor_G == []:
                data.circleColor_G = "green yellow"
            data.circleElem_G.append(circle(data.cirGrid[row][col],data.circleColor_G))

    #layer 3:B control by face
    for row in range(len(data.cirGrid)):
        for col in range(len(data.cirGrid[0])):
            data.cirGrid[row][col] = (data.cirGridLeft + data.cirGridSize * row, 
                data.cirGridTop + data.cirGridSize * col) 
            if  data.circleColor_B == []:
                data.circleColor_B = "cyan"
            data.circleElem_B.append(circle(data.cirGrid[row][col],data.circleColor_B))

    #data for square gird, 20 rows and 20 cols
    data.squareElem = []
    data.squGird = [([0] * 20) for row in range(50)]
    data.squGridSize = 40
    data.rotAngle = 0
    data.noise = 0
    data.radInc = True
    data.radDec = False
    #data.pxVal = [([0] * data.height) for row in range(data.width)]
    data.pxVal = [([0] * data.width) for row in range(data.height)]

    #initial square grid
    for row in range(len(data.squGird)):
        for col in range(len(data.squGird[0])):
            data.squGird[row][col] = (20 + data.squGridSize * row, 20 + data.squGridSize * col) 
            data.squareElem.append(rotatingSquare(data.squGird[row][col],data.rotAngle,data.noise))

    #for rectangle
    data.rectElem = []
    data.rectGird = [([0] * 25) for row in range(60)]
    data.rectHeight = 70
    data.rectWidth = 25
    data.rotFactor = 0
    for row in range(len(data.squGird)):
        for col in range(len(data.squGird [0])):
            #anchor points
            data.squGird [row][col] = (0 + data.rectWidth * row, 0 + data.rectHeight * col) 
            data.rectElem.append(rotatingRect(data.squGird [row][col],data.rotFactor))

def drawInterface(data,canvas):
	canvas.create_oval(data.width/2-100,data.height/2+60,data.width/2-100+15,data.height/2+60+15,fill = "black")
	canvas.create_rectangle(data.width/2,data.height/2+60,(data.width/2)+15,(data.height/2+60)+15,fill = "black")
	canvas.create_rectangle(data.width/2+100,data.height/2+55,(data.width/2+100)+15,(data.height/2+55)+30,fill = "black")
	canvas.create_text(data.width/2,data.height/2,text = "Press A / B / C to choose the mode to experoence")
	canvas.create_text(data.width/2,data.height/4,text="Metakinesis",fill="darkBlue",font="Helvetica 26 bold")
	canvas.create_text(data.width/2,data.height/2+20, text="A - circleIllusion / B - squareSwirl / C - rectangleWave")
	canvas.create_text(data.width/2,data.height/4 + 100, text="The project concept is to combine dancing with geometry algorithm to visualize the body movement.")
	canvas.create_text(data.width/2,data.height/4 + 120, text="To achieve this idea, I used OpenCV to analysis the dancer in the video and use those data to draw the moving pattern in Tkinter.")
    
    #press 1 - circleIllusion / 2 - squareSwirl / 3 - rectangleWave
    

def getPixelData(data):
    img = data.frame
    #get color of each pixel
    for i in range (len(data.pxVal)):
        for j in range (len(data.pxVal[0])):
            data.pxVal[i][j] = img[i,j]
            #print(i,j)
            # print(data.pxVal[i][j])
           
    
def outputFrame(canvas,data): 
    count = data.timer
    canvas.postscript(file="frame_%d.ps" %(count), colormode='color')
    fp = open("frame_%d.ps"%count, "rb")
    img = PIL.Image.open(fp)
    print(img.getdata)
    img.save("test",PNG)
    img.save("frame_%d.png" %(count), "png")

def timerFired(canvas,data):
    data.timer += 1
    boolCur, frame = data.cap.read()
    #store all frames into init
    data.frame = frame 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)

    data.faces = data.face_cascade.detectMultiScale(gray, 1.1 , 4)
    data.low = data.low_cascade.detectMultiScale(gray, 1.1 , 3)
    data.upper = data.upper_cascade.detectMultiScale(gray, 1.1 , 3)

    #update the circle center points (cx += dx,cy += dy) 
    #R:lower / G:upper / B:faces
    #update the circle size, control by the distance
    #R:close to the center points, circle bigger
    #G:close to the center points, circle smaller
    #B:close to the center points, circle bigger
    for circle in data.circleElem_R:
        if len(data.low) != 0:
            x = data.low[0][0]
            y = data.low[0][1]
            circle.move(data,x,y)
            circle.scale(data,x,y)
            circle.changeColor(x,y,1)

    for circle in data.circleElem_G:
        if len(data.upper) != 0:
            x = data.upper[0][0]
            y = data.upper[0][1]
            circle.move(data,x,y)
            circle.scale(data,x,y)
            circle.changeColor(x,y,2)

    for circle in data.circleElem_B:
        if len(data.faces) != 0:
            x = data.faces[0][0]
            y = data.faces[0][1]
            circle.move(data,x,y)
            circle.scale(data,x,y)
            circle.changeColor(x,y,3)

    #update the noise
    for square in data.squareElem:
        #angle control by lower body
        if len(data.low) != 0:
            if len(data.upper) != 0:
                data.rotAngle = data.low[0][1] * 0.01
                square.rotate(data,data.rotAngle)
                data.noise = (data.low[0][0],data.low[0][1])
                square.scale(data,data.noise)

            else:
                data.rotAngle = data.low[0][1] * 0.01
                square.rotate(data,data.rotAngle)
                data.noise = (data.low[0][0],data.low[0][1])
                square.scale(data,data.noise)
                
    #update the rect rotate angle
    for rect in data.rectElem:
        if len(data.low) != 0:
            #rotateFactor is lowerx position
            data.rotFactor = (data.low[0][0],data.low[0][1])
            rect.rotate(data,data.rotFactor)
            #rect.changeColor(data)

    #analyze the color in video
    getPixelData(data)
    #outputFrame(canvas,data)

def keyPressed(event, data):
    # use event.char and event.keysym
    if event.char == 'A':
        data.mode = 1
    elif event.char == 'B':
        data.mode = 2
    elif event.char == 'C':
        data.mode = 3
    print(event.keysym)
    print(data.mode)

def mousePressed(event, data):
    pass

def redrawAll(canvas, data):
    if data.mode == 0:
        drawInterface(data,canvas)
    elif data.mode == 1:
        #draw circle
        for cirGird in data.circleElem_R:
            cirGird.drawCircle(canvas)
        for cirGird in data.circleElem_G:
            cirGird.drawCircle(canvas)
        for cirGird in data.circleElem_B:
            cirGird.drawCircle(canvas)

    elif data.mode == 2:
        #draw square
        for square in data.squareElem :
            square.drawSquare(canvas)
    
    elif data.mode == 3:
        #draw rect
        for rect in data.rectElem :
            rect.drawRect(canvas)

####################################
# use the run function as-is
####################################

def run(width=800, height=800):

    def redrawAllWrapper(canvas, data):
        canvas.delete(ALL)
        canvas.create_rectangle(0, 0, data.width, data.height,
                                fill='white', width=0)
        redrawAll(canvas, data)
        canvas.update()    

    def mousePressedWrapper(event, canvas, data):
        mousePressed(event, data)
        redrawAllWrapper(canvas, data)

    def keyPressedWrapper(event, canvas, data):
        keyPressed(event, data)
        redrawAllWrapper(canvas, data)

    def timerFiredWrapper(canvas, data):
        timerFired(canvas, data)
        redrawAllWrapper(canvas, data)
        # pause, then call timerFired again
        canvas.after(data.timerDelay, timerFiredWrapper, canvas, data)
    # Set up data and call init
    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    data.timerDelay = 100 # milliseconds
    root = Tk()
    root.resizable(width=False, height=False) # prevents resizing window
    init(data)
    # create the root and the canvas
    canvas = Canvas(root, width=data.width, height=data.height)
    canvas.configure(bd=0, highlightthickness=0)
    canvas.pack()
    # set up events
    root.bind("<Button-1>", lambda event:
                            mousePressedWrapper(event, canvas, data))
    root.bind("<Key>", lambda event:
                            keyPressedWrapper(event, canvas, data))
    timerFiredWrapper(canvas, data)
    # and launch the app
    root.mainloop()  # blocks until window is closed
    print("bye!")
run(1280, 720)








