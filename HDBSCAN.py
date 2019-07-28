import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import sys
import scipy
import sklearn
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import random
import hdbscan

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
#HDBSCAN
hdbscanObj = hdbscan.HDBSCAN()
labels = hdbscanObj.fit_predict(coordinateVectors)
numParticles = max(labels) + 1 #Adding one because zero is a label
print("Number of germ plasm RNPs identified: " + str(numParticles)) 
HDBSCANSilhouette = sklearn.metrics.silhouette_score(coordinateVectors, labels)
print("HDBSCAN Silhouette score: " + str(HDBSCANSilhouette))
#Visualization 
#Generating a random list of colors for each label 
colors = generateColors(numParticles, labels)
#Creating a widget to view the clusters
createWidget(coordinateVectors,colors)