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
x = np.array(coordinates[0]*0.056, dtype = float); y = np.array(coordinates[1]*0.056, dtype = float); z = np.array(coordinates[2]*0.15, dtype = float)

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


#Checking if sorting data and labels worked 
colors = generateColors(numParticles, sortedLabels)
#Creating a widget for 3D plotting 
app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()
sp1 = gl.GLScatterPlotItem(pos=s, color = colors, pxMode=True, size = 0.0000001)
sp1.setGLOptions('opaque')
w.addItem(sp1)

'''
Data:
data - 4D data with x,y,z,label
sortedData - data sorted by label (has been denoised)
s - sorted list of points for plotting
sortedLabels - sorted labels, corresponds with sortedData and s
'''
    
#Finding the convex hull for every cluster
for i in range(0, int(numParticles),1):
    cluster = [j for j in sortedData if j[3] == i] #Accessing the points of every cluster
    c = [x[:-1] for x in cluster] #removing labels from cluster coordinates  
    
    #Checking if we have a 2D case or 3D case by checking min and max in each dimension 
    #Splitting x,y,z
    cx,cy,cz = zip(*c)
    #Getting the difference between the min and max element in each dimension 
    dx = max(cx) - min(cx);dy = max(cy) - min(cy);dz = max(cz) - min(cz)
    #Changing input to QHull depending on the dimension 
    if(dx == 0):
        input = np.vstack((y,z)).T
    elif(dy == 0):
        input = np.vstack((x,z)).T
    elif(dz == 0):
        input = np.vstack((x,y)).T
    else:
        input = c
    
    '''
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
    '''
    
    #Convex hull of the cluster
    convexHull = ConvexHull(input) 
    clusterVertices = convexHull.vertices
    #print(clusterVertices)
    
    #Making a list of coordinates of the vertices 
    vertices = list()
    for v in clusterVertices:
        vertices.append(input[v])
    
    #print(vertices)
    '''
    sp2 = gl.GLMeshItem(meshdata=vertices, color = [0,1,0,0])
    w.addItem(sp2)
    '''
    
    '''
    #plotting simplices (Source: https://stackoverflow.com/questions/27270477/3d-convex-hull-from-point-cloud)
    for s in convexHull.simplices:
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        ax.plot(c[s, 0], c[s, 1], c[s, 2], "r-")
    
    plt.show()
    '''

# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    QtGui.QApplication.instance().exec_() 
    