import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style
from statistics import mean
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d, Axes3D  
import time
import pandas as pd
from defs import EuclidDistance
from drawnow import *
import math

##### Initalizing Graphing Functions #####
 
X = []
Y = []
X2 = []
Y2 = []
Z =[]
fig = plt.figure()  # Create a figure and an axes.

plt.interactive(True) #Tell matplotlib you want interactive mode to plot live data


def makeFig(): #Create a function that makes our desired plot
    ax = Axes3D(fig)
    ax.plot(X,total_dist,Y, label='Target 1 #')         # X Y Z Axis
    ax.plot(X2,total_dist,Y2, label = 'Target 2 !')  
    ax.set_xlabel('Horzontal')
    ax.set_ylabel('Depth')
    ax.set_zlabel('Vertical')
    ax.set_title("Position")  # Add a title to the axes.
    ax.set_xlim(0,650)
    ax.set_zlim(0,400)
    ax.set_ylim(130,160)
    ax.legend(loc = 'upper right')  # Add a legend.
    plt.pause(.0001)                     #Pause Briefly. Important to keep drawnow from crashing
 
cap = cv2.VideoCapture(0)

template = cv2.imread('#2.png')
template_gray = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)

template2 = cv2.imread('!2.png')
template_gray2 = cv2.cvtColor(template2,cv2.COLOR_BGR2GRAY)

w,h = template_gray.shape[::-1]
w2,h2 = template_gray2.shape[::-1]

t = []
start = round(time.perf_counter(),1)
t.append(0)
BORDER = 2
THRES = 0.8
total_dist = []
PHYSICALDISTANCE = 4
xmove = 0
ymove = 0
x2move = 0
y2move = 0
zmove = 0

while cv2.waitKey(1) & 0xFF != ord('q'):
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    t.append(round(time.perf_counter(),1)-start)

    res1 = cv2.matchTemplate(gray_frame, template_gray, cv2.TM_CCOEFF_NORMED)   #TRY DIFFERENT METHODS
    loc = np.where(res1>=THRES)
    PossibleYs = loc[0].tolist()
    PossibleXs = loc[1].tolist()

    res2 = cv2.matchTemplate(gray_frame, template_gray2, cv2.TM_CCOEFF_NORMED)   #TRY DIFFERENT METHODS
    loc2 = np.where(res2>=THRES)
    PossibleY2s = loc2[0].tolist()
    PossibleX2s = loc2[1].tolist() 



    if len(PossibleY2s) != 0 and len(PossibleYs) != 0:              #Ensures BOTH targets are picked   
        x2 = PossibleX2s[-1]            # Picks last possible  
        y2 = PossibleY2s[-1]            # Picks last possible
        cv2.rectangle(frame, (x2,y2) , (x2 + w2, y2 + h2), (255,0,0),BORDER)
        x = PossibleXs[-1]              
        y = PossibleYs[-1]              
        cv2.rectangle(frame, (x,y), (x + w, y + h), (0,255,0),BORDER)

        xdist = x2-x
        ydist = y2-y
        EuclidDist = ((xdist**2)+(ydist**2))**(1/2)             # Euclidean Distance between two coordinates
        total_dist.append(EuclidDist)   

        cm_p = PHYSICALDISTANCE/EuclidDist

        half_angle = math.tan(31)
        M = cm_p*650
        z = (M/2)/half_angle

        if len(X) >=2:   
            xmove = (x - X[-1])*cm_p + xmove  
            ymove = (y - Y[-1])*cm_p + ymove
            x2move = (x2 - X2[-1])*cm_p + x2move  
            y2move = (y2 - Y2[-1])*cm_p + y2move
            zmove = z - Z[-1] + zmove
  
        X2.append(x2)
        Y2.append(y2)
        X.append(x)
        Y.append(y)
        Z.append(z)

        cv2.putText(frame,'Target1:  ' + str(round(xmove,3))+ ', ' + str(round(ymove,3)),(100,160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),2)
        cv2.putText(frame,'Target2:  ' + str(round(x2move,3))+ ', ' + str(round(y2move,3)),(100,180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),2)
        cv2.putText(frame, 'Depth:  ' + str(round(zmove,3)),(100,200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0),2)

        if cv2.waitKey(1) & 0xFF == ord('c'):           # Resets Displacement Values
            xmove = 0 
            ymove = 0
            x2move = 0
            y2move = 0
            zmove = 0




    cv2.imshow("frame",frame)
    # drawnow(makeFig)                       #Call drawnow to update our live graph

plt.close()
average_dist = mean(total_dist)
print("Average Distance: ", average_dist)
print("Time of reading: ", t[-1])
print('CM / Pixel: %.6f' %(PHYSICALDISTANCE/average_dist))

cv2.destroyAllWindows()
cap.release()