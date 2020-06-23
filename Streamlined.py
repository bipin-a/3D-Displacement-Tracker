import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style
from statistics import mean
from matplotlib.animation import FuncAnimation
import time
import pandas as pd
from defs import EuclidDistance

###### PLOTTING ############# 
X_AXISLIM = (0,650)
Y_AXISLIM = (0,400)

fig, ax1 = plt.subplots()
ax1.set_xlim(X_AXISLIM)
ax1.set_ylim(Y_AXISLIM)
ax1.set_xlabel('X-Axis')
ax1.set_ylabel('Y-Axis')
ax1.set_title('Window Displacement')
ax1.grid(True)  

cap = cv2.VideoCapture(0)

template = cv2.imread('#.png')
template_gray = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)

template2 = cv2.imread('!.png')
template_gray2 = cv2.cvtColor(template2,cv2.COLOR_BGR2GRAY)

w,h = template_gray.shape[::-1]
w2,h2 = template_gray2.shape[::-1]

X = []
Y = []
X2 = []
Y2 = []
t = []
start = round(time.perf_counter(),1)
t.append(0)
BORDER = 2
while True:
    # cv2.waitKey(20)
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    t.append(round(time.perf_counter(),1)-start)

    res = cv2.matchTemplate(gray_frame, template_gray, cv2.TM_CCOEFF_NORMED)   #TRY DIFFERENT METHODS
    THRES = 0.9
    loc = np.where(res>=THRES)
    Ys = loc[0].tolist()
    Xs = loc[1].tolist()
    if len(Xs) != 0:           # Can test either Xs or Ys, ensuring that object is detected
        x = Xs[-1]
        y = Ys[-1]
        X.append(x)
        Y.append(y)
        cv2.rectangle(frame, (x,y), (x + w, y + h), (0,255,0),BORDER)

    res = cv2.matchTemplate(gray_frame, template_gray2, cv2.TM_CCOEFF_NORMED)   #TRY DIFFERENT METHODS
    loc2 = np.where(res>=THRES)
    Y2s = loc2[0].tolist()
    X2s = loc2[1].tolist() 
    if len(X2s) != 0:           
        x2 = X2s[-1]
        y2 = Y2s[-1]
        X2.append(x2)
        Y2.append(y2)
        cv2.rectangle(frame, (x2,y2) , (x2 + w2, y2 + h2), (255,0,0),BORDER)

    cv2.imshow("frame",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break


PHYSICALDISTANCE = 4
avg_dist = EuclidDistance(X,Y,X2,Y2)
print("Average Distance: ", avg_dist)
print("Time of reading: ", t[-1])
print('CM / Pixel: %.2f' %(PHYSICALDISTANCE/avg_dist))


ax1.plot(X,Y, color='r')
ax1.plot(X2,Y2, color='b')

    

 
plt.tight_layout
plt.show()

cv2.destroyAllWindows()
cap.release()