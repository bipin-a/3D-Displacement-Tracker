from statistics import mean
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import style

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


def EuclidDistance(x,y,x2,y2):
    x0 = mean(x)
    y0 = mean(y)
    x1 = mean(x2)
    y1 = mean(y2)
    xdist = x1-x0
    ydist = y1-y0
    total_dist = ((xdist**2)+(ydist**2))**(1/2)  
    return (total_dist)

def beginPlot(X_AXISLIM, Y_AXISLIM, title):

    fig, ax1 = plt.subplots()
    ax1.set_xlim(X_AXISLIM)
    ax1.set_ylim(Y_AXISLIM)
    ax1.set_xlabel('X-Axis')
    ax1.set_ylabel('Y-Axis')
    ax1.set_title(title)
    ax1.grid(True) 

    return(ax1, fig)

def MovingAverage(loc1,loc2,windowsize):
    res = list(zip(*loc1))
    res2 = list(zip(*loc2))
    
    # Moving Average of the first Object
    numbers_series = pd.Series(res[0])
    windows = numbers_series.rolling(windowsize)
    moving_averages = windows.mean()
    moving_averages_list = moving_averages.tolist()
    X = moving_averages_list[windowsize - 1:]

    numbers_series = pd.Series(res[1])
    windows = numbers_series.rolling(windowsize)
    moving_averages = windows.mean()
    moving_averages_list = moving_averages.tolist()
    Y = moving_averages_list[windowsize - 1:]

    # Moving Average of the second Object
    numbers_series2 = pd.Series(res2[0])
    windows = numbers_series2.rolling(windowsize)
    moving_averages = windows.mean()
    moving_averages_list = moving_averages.tolist()
    X2 = moving_averages_list[windowsize - 1:]

    numbers_series2 = pd.Series(res2[1])
    windows = numbers_series2.rolling(windowsize)
    moving_averages = windows.mean()
    moving_averages_list = moving_averages.tolist()
    Y2 = moving_averages_list[windowsize - 1:]

    return(X,Y,X2,Y2)
