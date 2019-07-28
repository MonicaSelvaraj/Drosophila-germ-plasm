import numpy as np
import csv
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl

'''
Reads and stores the intensity matrix for one z-slice 
Cuts off first row and first column
plots  the intensity matrix
'''
def GetSliceIntensities(path):
	with open(path, 'r') as csv_file:
		matrix = []
		csv_reader = csv.reader (csv_file)
		for row in csv_reader:
			matrix.append(row)
		I = np.array(matrix)
		I = I[1:,1:]
		I = [[float(y) for y in x] for x in I]
		I = np.array(I)
		#plt.spy(I);plt.show()
	return I

'''
Returns x,y,z coordinates of fluorescence in current z-slice given the matrix of 
intensity values and which Z slice it is 
Temporary data (in micro meters): Pixel width,height: 0.0313030, Voxel depth: 0.1095510
'''
def getCoordinates(I,slice):
    y,x = I.nonzero() #y-rows, x-cols
    zlen = (len(y))
    z = [slice]*zlen
    x = np.array(x, dtype = float);y = np.array(y, dtype = float);z = np.array(z, dtype = float)
    return(x,y,z)

#Getting the coordinates for all the z-slices and storing it in an array called pos
pos = np.zeros((1,3)) #Making pos the same dimensions as the slice coordinates to allow for concatenation 
for i in range(0,11,1):
    I = GetSliceIntensities("Data/A_nos_embryo7_488_cmle-19-29/ZResults/Results"+str(i)+".csv") #I is the matrix of intensities
    AxisLim = I.shape[0]#Number of rows/columns
    x,y,z = getCoordinates(I,i)
    size = len(z)
    SlicePos = np.dstack((x,y,z))
    SlicePos = SlicePos[0]
    SlicePos = SlicePos.astype(int)
    pos = np.vstack((pos, SlicePos))
pos = np.delete(pos,0,0) #Deleting the [0,0,0] used for initialization
pos = pos.astype(int)

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

#Saving coordinates to file 
data = np.hsplit(pos,3)
X = np.array(data[0], dtype = int);Y = np.array(data[1], dtype = int);Z = np.array(data[2], dtype = int)
np.savetxt('3DCoordinates.csv', np.column_stack((X, Y, Z)), delimiter=",", fmt='%s')