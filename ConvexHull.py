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

#Lists to plot change in volume
dv = list();xv = list() #xv keeps track of which clusters had anomalous points removed 

for i in range(0, int(numParticles),1): #i iterates through the labels
#for i in range(0, 20,1):
    cluster = [j for j in denoisedData if j[3] == i] #Accessing the points of every cluster
    c = [x[:-1] for x in cluster] #removing labels from cluster coordinates  
    c = np.array(c, dtype = float)
    
    #Finding the convex hull of the original cluster
    oriCH = findConvexHull(c)
    if(oriCH == -1 or oriCH == 2): continue;
    oriVol = oriCH[2]
    
    cleanCluster = removeOutliers(c)
    
    #Finding the convex hull of the cluster after removing outliers
    cleanCH = findConvexHull(cleanCluster)
    if(cleanCH == -1 or cleanCH == 2): continue;
    cleanVol = cleanCH[2]
    
    if((oriVol - cleanVol) != 0):xv.append(i);dv.append(oriVol - cleanVol)

dvMean = np.mean(dv)
dvSd = np.std(dv)
dvMedian = np.median(dv)
dvMedDev = scipy.stats.median_absolute_deviation(dv)
volThreshold = dvMedian + 2*dvMedDev

print('')
print('Data')
print('Number of clusters: ' + str(numParticles))
print('Mean change in volume: ' + str(dvMean))
print('Standard deviation of change in volume: ' + str(dvSd))
print('Median change in volume: ' + str(dvMedian))
print('Median absolute deviation of change in volume: ' + str(dvMedDev))
print('Change in volume threshold: ' + str(volThreshold))

clustersWithOutliers = list() #Since volume threshold is not scaled, keeping track of the clusters with outliers so they can be removed later in the script

for v in range(0, len(dv), 1):
    if(dv[v] > (dvMedian + 2*dvMedDev)): plt.scatter(xv[v],dv[v],c='g',s=10);clustersWithOutliers.append(xv[v])
    else: plt.scatter(xv[v],dv[v],c='b',s=10)
plt.xlabel('Cluster number')
plt.ylabel('Change in volume')
#plt.xlim(0, 150);plt.ylim(0, 150)
plt.show()

print('Clusters with outliers: ' + str(clustersWithOutliers))
print('Number of clusters with outliers: ' + str(len(clustersWithOutliers)))


'''
Using the change in volume threshold to remove anomalous points from clusters
and finding the total volume
'''
#Scaling coordinates 
xs = np.array(coordinates[0]*0.056, dtype = float); ys = np.array(coordinates[1]*0.056, dtype = float); zs = np.array(coordinates[2]*0.15, dtype = float)
#Making tuples of the form (x,y,z,label)
dataScaled = np.vstack((xs,ys,zs,labels)).T
#Removing noise points
denoisedDataScaled = [i for i in dataScaled if i[3] > -1]

totalVol = 0

for i in range(0, int(numParticles),1): #i iterates through the labels
#for i in range(0, 20,1):
    cluster = [j for j in denoisedDataScaled if j[3] == i] #Accessing the points of every cluster
    c = [x[:-1] for x in cluster] #removing labels from cluster coordinates  
    c = np.array(c, dtype = float)
    
    if i in clustersWithOutliers:
        cleanCluster = removeOutliers(c)
        cleanCH = findConvexHull(cleanCluster)
        if(cleanCH == -1 or cleanCH == 2): continue;
        totalVol = totalVol + cleanCH[2]
    else:
        oriCH = findConvexHull(c)
        if(oriCH == -1 or oriCH == 2): continue;
        totalVol = totalVol + oriCH[2]

print('Total scaled volume after anomaly detection: ' + str(totalVol))
print('')

    