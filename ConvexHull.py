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

def getList(filename):
    x = list()
    with open (filename, 'r') as csv_file:
        csv_reader = csv.reader (csv_file)
        for line in csv_reader:
        	x.append(line[0])
    x = np.array(x, dtype = float)
    return (x)


#Main
#Getting pixel coordinates
coordinates = getPoints('3DCoordinates.csv')

#Scaling point cloud to biological size in microns
#Voxel size for A_nos_embryo7_488_cmle-19-29 is 0.056 x 0.056 x 0.15 micron^3
x = np.array(coordinates[0]*0.056, dtype = float); y = np.array(coordinates[1]*0.056, dtype = float); z = np.array(coordinates[2]*0.15, dtype = float)

#Getting labels 
labels = getList('hdbscanLabels.csv')
numClusters = max(labels) + 1 

#Making tuples of the form (x,y,z,label)
data = np.vstack((x,y,z,labels)).T

#Removing noise points
denoisedData = [i for i in data if i[3] > -1]

#Sorting by label 
sortedData =  sorted(denoisedData, key=lambda tup: tup[3])

#Finding the convex hull for every cluster
#for i in range(0, int(numClusters),1):
for i in range(100, 101,1):
    cluster = [j for j in sortedData if j[3] == i] #Accessing the points of every cluster 
    c = [x[:-1] for x in cluster] #removing labels from coordinates 
    c= np.array(c)
    print(c)
        
    #Visualizing the cluster
    xc = list(); yc = list(); zc = list()
    for p in c:
        	xc.append(p[0]); yc.append(p[1]); zc.append(p[2])
    fig = plt.figure( )
    plt.style.use('dark_background')
    ax = fig.add_subplot(1,1,1, projection = '3d')
    ax.grid(False)
    ax.scatter (xc,yc,zc, c = 'g', marker='o', s=100, linewidths=2)
    ax.set_title('Visualizing one cluster')
    ax.set_xlabel ('x, axis')
    ax.set_ylabel ('y axis')
    ax.set_zlabel ('z axis')
    
    #Convex hull of the cluster
    convexHull = ConvexHull(c) 
    #plotting simplices (Source: https://stackoverflow.com/questions/27270477/3d-convex-hull-from-point-cloud)
    for s in convexHull.simplices:
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        ax.plot(c[s, 0], c[s, 1], c[s, 2], "r-")
    
    plt.show()


'''
#Making lists of points that belong to the same cluster
coordinateVectors = np.vstack((x,y,z)).T
print(coordinateVectors)
#Sorting points by label 
labels = getList('hdbscanLabels.csv')
sortedCoordinates = [x for _,x in sorted(zip(labels,coordinateVectors))]
print(sortedCoordinates)

#Finding the convex hull of each cluster

#Finding the volume occupied by the convex hull of each cluster

#Checking if the scaling is correct
pos = np.vstack((x,y,z)).T

#Creating a widget for 3D plotting 
app = QtGui.QApplication([])
w = gl.GLViewWidget()
#w = gl.setGLOptions('opaque')
w.show()
sp2 = gl.GLScatterPlotItem(pos=pos, color = [0,1,0,1], pxMode=True, size = 0.0000001)
sp2.setGLOptions('opaque')
w.addItem(sp2)
# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    QtGui.QApplication.instance().exec_()
'''
