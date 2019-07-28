import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import sys
import scipy
import scipy.sparse as sparse
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import random

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

#Main
#Getting pixel coordinates
pos = getPoints('PixelCoordinates.csv')

#Checking out labeling of regions of fluorescence (RNP labeling) using DBSCAN
coordinateVector = np.vstack((pos[0],pos[1],pos[2])).T

#kNN distance graph to determine epsilon for DBSCAN
#Source - https://scikit-learn.org/stable/modules/neighbors.html
nbrs = NearestNeighbors(n_neighbors=8).fit(coordinateVector)
distances, indices = nbrs.kneighbors(coordinateVector)
sortedDistancesInc = sorted(distances[:,6],reverse=False)
plt.plot(list(range(1,len(pos[0])+1)), sortedDistancesInc)
plt.show()

#By eye estimate - an epsilon of 1.5 looks good for 8NN

clustering = DBSCAN(eps=1.5, min_samples=8).fit(coordinateVector)
labels = clustering.labels_
np.set_printoptions(threshold=np.inf)
numParticles = max(labels) + 1 #Adding one because zero is a label
print("Number of germ plasm RNP's identified: " + str(numParticles)) 

#Visualization 
#Generating a random list of colors for each label 

color = list()
random.seed() #Initializing the random number generator 
randomColors = [ ( random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1) ) for i in range(0,numParticles) ]
for label in labels:
	if(label == -1):color.append((0,0,0,0)) #Assigning black to noise/non-granules
	else: color.append(randomColors[label])
color = np.array(color, dtype = float)

#Creating a widget to view the clusters
app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()
sp2 = gl.GLScatterPlotItem(pos=coordinateVector, color = color, pxMode=True, size = 0.0000001)
sp2.setGLOptions('opaque')
w.addItem(sp2)
# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    QtGui.QApplication.instance().exec_()