import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import sys
import scipy
#from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.axes3d import Axes3D
import random
from sklearn.ensemble import IsolationForest
import numpy.matlib

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
Use: Finds convex hull of a set of points - deals with cases that throws exceptions 
returns convex hull dimension, simplices, and volume
'''
def findConvexHull(c):
    dim=-1
    #Case 1: Very small clusters 
    if(len(c) <= 3): return dim #print('Cluster with ' + str(len(c)) + 'points'); return dim
    #Case 2: Flat/2D clusters
    cx,cy,cz = zip(*c) 
    dx = max(cx) - min(cx);dy = max(cy) - min(cy);dz = max(cz) - min(cz)
    if(dx == 0 or dy == 0 or dz == 0): 
        #print('Flat cluster')
        dim = 2
        return dim #Figure out how to deal with 2D case
        if(dx == 0):input = np.vstack((cy,cz)).T
        elif(dy == 0):input = np.vstack((cx,cz)).T
        elif(dz == 0):input = np.vstack((cx,cy)).T
        else:input = c
    #Case 3: cluster in 3D
    dim = 3
    input = c
    #Computing convex hull
    convexHull = ConvexHull(input)
    return (dim,convexHull.simplices,convexHull.volume)

'''
Use: Detects and removes outliers in a 3D point cloud using isolation forests 
'''
def removeOutliers(c):
    model = IsolationForest(behaviour="new",max_samples=len(c),contamination='auto',n_estimators=1000)
    model.fit(c)
    sklearn_score_anomalies = model.decision_function(c)
    original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies] #(Source: https://stats.stackexchange.com/questions/335274/scikit-learn-isolationforest-anomaly-score)
    meanScore = np.mean(original_paper_score)
    sdScore = np.std(original_paper_score)
    cleanCluster = list()
    for i in range(0, len(c), 1):
        if(original_paper_score[i]>(meanScore+2*sdScore)): continue
        else: cleanCluster.append(c[i])
    cleanCluster = np.array(cleanCluster, dtype = float)
    return cleanCluster

def getVolThreshold(x,y):
    #The elbow point is the point on the curve with the maximum absolute second derivative 
    #Source: https://dataplatform.cloud.ibm.com/analytics/notebooks/54d79c2a-f155-40ec-93ec-ed05b58afa39/view?access_token=6d8ec910cf2a1b3901c721fcb94638563cd646fe14400fecbb76cea6aaae2fb1
    data = np.vstack((x,y)).T
    nPoints = len(x)
    #Drawing a line from the first point to the last point on the curve 
    firstPoint = data[0]
    lastPoint = data[-1]
    lv = lastPoint - firstPoint #Finding a vector between the first and last point
    lvn = lv/np.linalg.norm(lv)#Normalizing the vector
    #Finding the distance to the line 
    vecFromFirst = data - firstPoint
    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lvn, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lvn)
    vecToLine = vecFromFirst - vecFromFirstParallel
    # distance to line is the norm of vecToLine
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    
    # knee/elbow is the point with max distance value
    idxOfBestPoint = np.argmax(distToLine)
    
    plt.scatter(x, y, c = 'b')
    plt.scatter(firstPoint[0],firstPoint[1], c='r')
    plt.scatter(lastPoint[0],lastPoint[1], c='r')
    plt.plot([firstPoint[0],lastPoint[0]],[firstPoint[1],lastPoint[1]])
    plt.scatter(data[idxOfBestPoint][0],data[idxOfBestPoint][1])
    plt.xlabel('Cluster number')
    plt.ylabel('Change in volume')
    plt.show()
    return data[idxOfBestPoint][1]

#Main
#Getting pixel coordinates
coordinates = getPoints('3DCoordinates.csv')
x = np.array(coordinates[0], dtype = float); y = np.array(coordinates[1], dtype = float); z = np.array(coordinates[2], dtype = float)
#Getting labels 
labels = getList('hdbscanLabels.csv')
numParticles = int(max(labels) + 1)
#Making tuples of the form (x,y,z,label)
data = np.vstack((x,y,z,labels)).T
#Removing noise points
denoisedData = [i for i in data if i[3] > -1]
#Removing anomalous points from a cluster if present
#Lists to plot change in volume
dv = list();xv = list() #xv is cluster number/x axis for the volume plot 
vB = list(); vA = list();vC = list() #Storing volume before, after, and change in lists for easily calculating volume after finding the threshold 
for i in range(0, int(numParticles),1): #i iterates through the labels
#for i in range(0, 20,1):
    cluster = [j for j in denoisedData if j[3] == i] #Accessing the points of every cluster
    c = [x[:-1] for x in cluster] #removing labels from cluster coordinates  
    c = np.array(c, dtype = float)
    
    #Finding the convex hull of the original cluster
    oriCH = findConvexHull(c)
    if(oriCH == -1 or oriCH == 2): continue;
    oriVol = oriCH[2]
    vB.append(oriVol)
    
    cleanCluster = removeOutliers(c)
    
    #Finding the convex hull of the cluster after removing outliers
    cleanCH = findConvexHull(cleanCluster)
    if(cleanCH == -1 or cleanCH == 2): continue;
    cleanVol = cleanCH[2]
    vA.append(cleanVol)
    
    vC.append(oriVol - cleanVol)
    
    if((oriVol - cleanVol) != 0):xv.append(i);dv.append(oriVol - cleanVol)

dvMean1 = np.mean(dv);dvSd1 = np.std(dv)
dvMedian1 = np.median(dv)
medDev = scipy.stats.median_absolute_deviation(dv)
print('')
print('Data')
print('Mean change in volume: ' + str(dvMean1))
print('Standard deviation of change in volume: ' + str(dvSd1))
print('Median change in volume: ' + str(dvMedian1))
print('Median absolute deviation of change in volume: ' + str(medDev))
for v in range(0, len(dv), 1):
    if(dv[v] > (dvMedian1 + 2*medDev)): plt.scatter(xv[v],dv[v],c='g',s=10)
    else: plt.scatter(xv[v],dv[v],c='b',s=10)
plt.xlabel('Cluster number')
plt.ylabel('Change in volume')
plt.xlim(0, 150);plt.ylim(0, 150)
#plt.text(80, 80, r'$\mu=\dvMean1,\ \sigma=dvSd1$')
plt.show()

'''
for v in range(0, len(dv), 1):
    if(dv[v] > (dvMean1 + 2*dvSd1)): plt.scatter(xv[v],dv[v],c='g')
    else: plt.scatter(xv[v],dv[v],c='b')
plt.show()
'''

dv.sort()
#Generating a list of numbers from 1 to len(dv)
xaxis = list(range(0,len(dv)))
#Calculating total volume
#volThreshold = getVolThreshold(xaxis,dv)
volThreshold = dvMedian1 + 2*medDev
print('Change in volume threshold: ' + str(volThreshold))
    
totalVol = 0
for m in range(0,len(vC),1):
    if(vC[m]>=volThreshold): totalVol = totalVol + vA[m]
    else: totalVol = totalVol + vB[m]
print('Total volume after anomaly detection: ' + str(totalVol))
print('')
    
    