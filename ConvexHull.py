import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import sys
import scipy
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl


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
coordinates = getPoints('3DCoordinates.csv')
#Scaling point cloud to biological size in microns
#Voxel size for A_nos_embryo7_488_cmle-19-29 is 0.056 x 0.056 x 0.15 micron^3
x = np.array(coordinates[0]*0.056, dtype = float); y = np.array(coordinates[1]*0.056, dtype = float); z = np.array(coordinates[2]*0.15, dtype = float)

'''
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
