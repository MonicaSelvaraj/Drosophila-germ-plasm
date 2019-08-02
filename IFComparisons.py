import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import sys
import scipy
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.axes3d import Axes3D
import random
from sklearn.ensemble import IsolationForest

'''
Use: To read a file with x,y,z coordinates, and store the data for each dimension in a separate array.
params: filename - File with x,y,z cooridnates
returns: 3 arrays with x's, y's and z's
'''
def getPoints(filename):
    x = list(); y = list(); z = list()
    with open (filename, 'r') as csv_file:
        csv_reader = csv.reader (csv_file)
        for line in csv_reader:
        	x.append(line[0]); y.append(line[1]); z.append(line[2])
    x = np.array(x, dtype = float); y = np.array(y, dtype = float); z = np.array(z, dtype = float)
    return (x, y, z)
'''
Use: To read a file with one dimensional data and store it in an array 
'''
def getList(filename):
    x = list()
    with open (filename, 'r') as csv_file:
        csv_reader = csv.reader (csv_file)
        for line in csv_reader:
        	x.append(line[0])
    x = np.array(x, dtype = float)
    return (x)

'''
Use: Generate a random list of colors and assign colors to coordinates based on which cluster it belongs to.
'''
def generateColors(numParticles, labels):
    colors = list()
    random.seed() #Initializing the random number generator 
    randomColors = [ ( random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1) ) for i in range(0,numParticles) ]
    for label in labels:
        if(label == -1):colors.append((0,0,0,0)) #Assigning black to noise/non-granules
        else: colors.append(randomColors[label])
    colors = np.array(colors, dtype = float)
    return colors

#Main
#Getting pixel coordinates
coordinates = getPoints('3DCoordinates.csv')

#Scaling point cloud to biological size in microns
#Voxel size for A_nos_embryo7_488_cmle-19-29 is 0.056 x 0.056 x 0.15 micron^3
#x = np.array(coordinates[0]*0.056, dtype = float); y = np.array(coordinates[1]*0.056, dtype = float); z = np.array(coordinates[2]*0.15, dtype = float)
x = np.array(coordinates[0], dtype = float); y = np.array(coordinates[1], dtype = float); z = np.array(coordinates[2], dtype = float)

#Getting labels 
labels = getList('hdbscanLabels.csv')
numParticles = int(max(labels) + 1)

#Making tuples of the form (x,y,z,label)
data = np.vstack((x,y,z,labels)).T

#Removing noise points
denoisedData = [i for i in data if i[3] > -1]

#Sorting by label 
sortedData =  sorted(denoisedData, key=lambda tup: tup[3])

s = [x[:-1] for x in sortedData] #Remove after checking for plotting 
s = np.stack( s, axis=0 )

#Removing noise from labels and sorting - for plotting 
denoisedLabels =  [i for i in labels if i > -1]
denoisedLabels.sort() #sorting labels to match sorted Data for plotting 
sortedLabels = [int(i) for i in denoisedLabels]

'''
#Checking if sorting data and labels worked 
colors = generateColors(numParticles, sortedLabels)
#Creating a widget for 3D plotting 
app = QtGui.QApplication([])
w = gl.GLViewWidget()
#w.show()
sp1 = gl.GLScatterPlotItem(pos=s, color = colors, pxMode=True, size = 0.0000001)
sp1.setGLOptions('opaque')
w.addItem(sp1)
'''

'''
Data:
data - 4D data with x,y,z,label
sortedData - data sorted by label (has been denoised)
s - sorted list of points for plotting
sortedLabels - sorted labels, corresponds with sortedData and s
'''

rem = 0
#Finding the convex hull for every cluster
for i in range(0, int(numParticles),1):
#for i in range(0,50,1):
    print(i)
    cluster = [j for j in sortedData if j[3] == i] #Accessing the points of every cluster
    c = [x[:-1] for x in cluster] #removing labels from cluster coordinates  
    c = np.array(c, dtype = float)
    
    #Removing very small clusters here
    if(len(c) <= 3): print('Removed cluster with' + str(len(c)) + 'points'); continue;
    
    '''
    Finding the convex hull of clusters without removing outliers 
    '''
    #Checking if we have a 2D case or 3D case by checking min and max in each dimension 
    cx,cy,cz = zip(*c) 
    dx = max(cx) - min(cx);dy = max(cy) - min(cy);dz = max(cz) - min(cz)#Getting the difference between the min and max element in each dimension
    if(dx == 0 or dy == 0 or dz == 0): print('Flat cluster');continue
    
    
    
    #scaling hull input after removing outliers
    scx,scy,scz = zip(*c)
    scx = np.array(scx, dtype = float); scy = np.array(scy, dtype = float); scz = np.array(scz, dtype = float)
    scx = scx*0.056;scy = scy*0.056;scz = scz*0.15
    #scx = np.array(scx*0.056, dtype = float); scy = np.array(scy*0.056, dtype = float); scz = np.array(scz*0.15, dtype = float)
    scaledHullInput = np.vstack((scx,scy,scz)).T

    
    convexHull = ConvexHull(scaledHullInput)
    
    '''
    Finding the convex hull of clusters after removing outliers
    '''
    #Removing anomalous points from the cluster
    model = IsolationForest(behaviour="new",max_samples=len(c),contamination='auto',n_estimators=1000)
    model.fit(c)
    sklearn_score_anomalies = model.decision_function(c)
    original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies] #(Source: https://stats.stackexchange.com/questions/335274/scikit-learn-isolationforest-anomaly-score)
    print('Anomaly scores')
    print(original_paper_score)
    meanScore = np.mean(original_paper_score)
    print('Mean score: ' + str(meanScore))
    sdScore = np.std(original_paper_score)
    print('Standard deviation: ' + str(sdScore))
    print('Mean + 1*sd: '+ str(meanScore+sdScore))
    print('Mean + 2*sd: '+ str(meanScore+2*sdScore))
    cleanCluster = list()
    for i in range(0, len(c), 1):
        if(original_paper_score[i]>(meanScore+2*sdScore)): continue
        else: cleanCluster.append(c[i])
    cleanCluster = np.array(cleanCluster, dtype = float)
    
    ccx,ccy,ccz = zip(*cleanCluster) 
    cdx = max(ccx) - min(ccx);cdy = max(ccy) - min(ccy);cdz = max(ccz) - min(ccz)#Getting the difference between the min and max element in each dimension
    if(cdx == 0 or cdy == 0 or cdz == 0): print('Flat cluster');continue
    
    '''
    Set up for 3d plotting 
    '''
    fig = plt.figure( )
    plt.style.use('dark_background')
    #plotting before here
    ax = fig.add_subplot(1,2,1, projection = '3d')
    ax.grid(False)
    #Visualizing the cluster
    ax.scatter (scx,scy,scz, c = 'g', marker='o', s=10, linewidths=2)
    ax.set_title('Before')
    ax.set_xlabel ('x, axis')
    ax.set_ylabel ('y axis')
    ax.set_zlabel ('z axis')
    #plotting simplices (Source: https://stackoverflow.com/questions/27270477/3d-convex-hull-from-point-cloud)
    for s in convexHull.simplices:
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        ax.plot(scaledHullInput[s, 0], scaledHullInput[s, 1], scaledHullInput[s, 2], "r-")
    
    print('Before:');print(len(c))
    print('After');print(len(cleanCluster))
    
    
    if(len(c) - len(cleanCluster) != 0): rem = rem+1
    
    #Scaling input 
    sccx,sccy,sccz = zip(*cleanCluster)
    sccx = np.array(sccx, dtype = float); sccy = np.array(sccy, dtype = float); sccz = np.array(sccz, dtype = float)
    sccx = sccx*0.056;sccy = sccy*0.056;sccz = sccz*0.15
    #sccx = np.array(sccx*0.056, dtype = float); sccy = np.array(sccy*0.056, dtype = float); sccz = np.array(sccz*0.15, dtype = float)
    scaledHullInputClean = np.vstack((sccx,sccy,sccz)).T
    
    convexHull = ConvexHull(scaledHullInputClean) 
    
    ax = fig.add_subplot(1,2,2, projection = '3d')
    ax.grid(False)
    #Visualizing the cluster
    ax.scatter (sccx,sccy,sccz, c = 'g', marker='o', s=10, linewidths=2)
    ax.set_title('After')
    ax.set_xlabel ('x, axis')
    ax.set_ylabel ('y axis')
    ax.set_zlabel ('z axis')
    #plotting simplices (Source: https://stackoverflow.com/questions/27270477/3d-convex-hull-from-point-cloud)
    for s in convexHull.simplices:
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        ax.plot(scaledHullInputClean[s, 0], scaledHullInputClean[s, 1], scaledHullInputClean[s, 2], "r-")
    plt.show()
    
print('Points removed from ' + str(rem) + ' clusters')
'''
# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    QtGui.QApplication.instance().exec_() 
'''