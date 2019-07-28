import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import sys
import scipy
import scipy.sparse as sparse
import sklearn
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import random
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
Use: Automatically find epsilon for DBSCAN from a kNN distance graph 
'''
def getEpsilon(k,coordinateVectors):
    #Using a kNN distance graph to determine epsilon for DBSCAN (Source - https://scikit-learn.org/stable/modules/neighbors.html)
    nbrs = NearestNeighbors(n_neighbors=k).fit(coordinateVectors)
    distances, indices = nbrs.kneighbors(coordinateVectors)
    sortedDistancesInc = sorted(distances[:,k-1],reverse=False) #sorting distances in ascending order
    #plt.plot(list(range(1,len(coordinateVectors)+1)), sortedDistancesInc)
    #plt.show()
    
    #Automatically obtaining epsilon from the kNN distance plot
    #The elbow point is the point on the curve with the maximum absolute second derivative 
    #Source: https://dataplatform.cloud.ibm.com/analytics/notebooks/54d79c2a-f155-40ec-93ec-ed05b58afa39/view?access_token=6d8ec910cf2a1b3901c721fcb94638563cd646fe14400fecbb76cea6aaae2fb1
    x = list(range(1,len(coordinateVectors)+1))
    y = sortedDistancesInc
    kNNdata = np.vstack((x,y)).T
    nPoints = len(x)
    #Drawing a line from the first point to the last point on the curve 
    firstPoint = kNNdata[0]
    lastPoint = kNNdata[-1]
    #plt.scatter(firstPoint[0],firstPoint[1], c='blue',s=10)
    #plt.scatter(lastPoint[0],lastPoint[1], c='blue',s=10)
    lv = lastPoint - firstPoint #Finding a vector between the first and last point
    lvn = lv/np.linalg.norm(lv)#Normalizing the vector
    #plt.plot([firstPoint[0],lastPoint[0]],[firstPoint[1],lastPoint[1]])
    #plt.show()
    
    #Finding the distance to the line 
    vecFromFirst = kNNdata - firstPoint
    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lvn, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lvn)
    vecToLine = vecFromFirst - vecFromFirstParallel
    # distance to line is the norm of vecToLine
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    
    # knee/elbow is the point with max distance value
    idxOfBestPoint = np.argmax(distToLine)
    
    #print ("Knee of the curve is at index =",idxOfBestPoint)
    #print ("Knee value =", kNNdata[idxOfBestPoint])
    return kNNdata[idxOfBestPoint]

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

'''
Use: To create a GUI for 3D point cloud visualization
'''
def createWidget(coordinateVectors,colors):
    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.show()
    sp = gl.GLScatterPlotItem(pos=coordinateVectors, color = colors, pxMode=True, size = 0.0000001)
    sp.setGLOptions('opaque')
    w.addItem(sp)
    # Start Qt event loop unless running in interactive mode.
    if __name__ == '__main__':
        QtGui.QApplication.instance().exec_()
    return

#Main
#Getting pixel coordinates
coordinates = getPoints('3DCoordinates.csv')
#coordinateVectors is the input to clustering algorithms 
coordinateVectors = np.vstack((coordinates[0],coordinates[1],coordinates[2])).T
epsilon = getEpsilon(6, coordinateVectors)
epsilon = epsilon[1]
#DBSCAN
clustering = DBSCAN(eps=epsilon, min_samples=6).fit(coordinateVectors)
labels = clustering.labels_
#np.set_printoptions(threshold=np.inf)
numParticles = max(labels) + 1 #Adding one because zero is a label
print("Number of germ plasm RNPs identified: " + str(numParticles)) 
DBSCANSilhouette = sklearn.metrics.silhouette_score(coordinateVectors, labels)
print("DBSCAN Silhouette score: " + str(DBSCANSilhouette))
#Visualization 
#Generating a random list of colors for each label 
colors = generateColors(numParticles, labels)
#Creating a widget to view the clusters
createWidget(coordinateVectors,colors)