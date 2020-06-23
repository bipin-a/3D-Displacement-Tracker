import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style
import time
import pandas as pd
from defs import EuclidDistance
from defs import MovingAverage

cap = cv2.VideoCapture(0)

template = cv2.imread('#.png')
template_gray = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)

template2 = cv2.imread('!.png')
template_gray2 = cv2.cvtColor(template2,cv2.COLOR_BGR2GRAY)

w,h = template_gray.shape[::-1]
w2,h2 = template_gray2.shape[::-1]

locations = []
locations2 = []
t = []
start = round(time.perf_counter(),1)
t.append(0)
THICK = 2


while True:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    t.append(round(time.perf_counter(),1)-start)
    # gray_frame = np.float32(gray_frame)
    dst = cv2.cornerHarris(gray_frame, 2, 5, 0.2)
 
    dst = cv2.dilate(dst, None)

    frame[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv2.imshow("Harris Corner",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

''' 
WINSIZE = 9

# SMOOTHES COORDINATES (TRY COMMENTING OUT TO IMPROVE SPEED)
X,Y,X2,Y2 = MovingAverage(locations,locations2,WINSIZE)

PHYSICALDISTANCE = 4
avg_dist = EuclidDistance(X,Y,X2,Y2)
print("Average Distance: ", avg_dist)
print("Length of # Symbol: ", len(X))
print("Length of ! Symbol: ", len(X2))
print("Time of reading: ", t[-1])
###### PLOTTING ############# 
X_AXISLIM = (0,650)
Y_AXISLIM = (0,400)

fig, ax1 = plt.subplots()

ax1.plot(X,Y, color='r')
ax1.plot(X2,Y2, color='b')
ax1.set_xlim(X_AXISLIM)
ax1.set_ylim(Y_AXISLIM)
ax1.set_xlabel('X-Axis')
ax1.set_ylabel('Y-Axis')

ax1.set_title('Moving Average Window Displacement')
ax1.grid(True)

plt.tight_layout
plt.show()

'''

cv2.destroyAllWindows()
cap.release()