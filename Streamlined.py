import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style
from statistics import mean
from matplotlib.animation import FuncAnimation
import time
import pandas as pd
from dependencies import beginPlot
from dependencies import EuclidDistance

class Target: 
    def __init__(self, img):
        self.image = img
        self.greyImage = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        self.width, self.height = self.greyImage.shape[::-1]
        self.yList = []
        self.xList = []

    def addCoordinates(self,x,y):
        self.xCoordinate = x
        self.yCoordinate = y        
        self.xList.append(x)
        self.yList.append(y)

    def DrawOnFrame (self, frame, borderSize, color = (0,255,0)):
        cv2.rectangle(frame, (self.xCoordinate,self.yCoordinate), 
        (self.xCoordinate + self.width, self.yCoordinate + self.height) ,color,borderSize)


# Plotting 
X_AXISLIM = (0,650)
Y_AXISLIM = (0,400)
ax1,fig = beginPlot(X_AXISLIM, Y_AXISLIM, title = "Displacement")

# Delcaring constants
BORDER = 2
THRES = 0.8
PHYSICALDISTANCE = 4

cap = cv2.VideoCapture(0)   # Begin capturing video from default camera source

# Instantiating two target objects
template = Target(img=cv2.imread('#.png'))
template2 = Target(img= cv2.imread('!.png'))

# Begin Timer
t = []
start = round(time.perf_counter(),1)
t.append(0)


while True:
    # cv2.waitKey(20)
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    t.append(round(time.perf_counter(),1)-start)

    # Matching first target
    results = cv2.matchTemplate(gray_frame, template.greyImage, cv2.TM_CCOEFF_NORMED)  
    locations = np.where(results>=THRES)
    Ys = locations[0].tolist()
    Xs = locations[1].tolist()
    if len(Xs) != 0:           # Can test either Xs or Ys, ensuring that object is detected
        template.addCoordinates(x = Xs[-1], y = Ys[-1])
        template.DrawOnFrame(frame=frame,borderSize= BORDER)

    # Matching second target
    results2 = cv2.matchTemplate(gray_frame, template2.greyImage, cv2.TM_CCOEFF_NORMED) 
    locations2 = np.where(results2>=THRES)
    Y2s = locations2[0].tolist()
    X2s = locations2[1].tolist() 
    if len(X2s) != 0:           
        template2.addCoordinates(x = X2s[-1], y = Y2s[-1])
        template2.DrawOnFrame(frame=frame,borderSize= BORDER, color= (255,255,0))

    
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break


# Output 
avg_dist = EuclidDistance(template.xList, template.yList, template2.xList, template2.yList)
print("Average Distance: ", avg_dist)
print("Time of reading: ", t[-1])
print('CM / Pixel: %.2f' %(PHYSICALDISTANCE/avg_dist))


ax1.plot(template.xList,template.yList, color='r')
ax1.plot(template2.xList,template2.yList, color='b') 
plt.tight_layout
plt.show()

cv2.destroyAllWindows()
cap.release()

