'''
## build a QApplication before building other widgets
import pyqtgraph as pg
pg.mkQApp()
## make a widget for displaying 3D objects
import pyqtgraph.opengl as gl
view = gl.GLViewWidget()
view.show()
## create three grids, add each to the view
xgrid = gl.GLGridItem()
ygrid = gl.GLGridItem()
zgrid = gl.GLGridItem()
view.addItem(xgrid)
view.addItem(ygrid)
view.addItem(zgrid)
## rotate x and y grids to face the correct direction
xgrid.rotate(90, 0, 1, 0)
ygrid.rotate(90, 1, 0, 0)
data = [1,2,3,4,5,6]
pg.plot(data)
_ = loop.with_bg_task(plot.update, plot.save).run()  # run the loop
'''

'''
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
app = QtGui.QApplication([])
mw = QtGui.QMainWindow()
view = pg.GraphicsLayoutWidget()  
mw.setCentralWidget(view)
mw.setWindowTitle('pyqtgraph example: ScatterPlot')
w1 = view.addPlot()
x = [1,2,3,4,5,6,7,8,9,10]
y = [10,8,6,4,2,20,18,16,14,12]
s1 = pg.ScatterPlotItem(x,y,size=10)
w1.addItem(s1)
mw.show()
'''
'''
def Start():
    m = myWindow()
    m.show()
    return m
#class myWindow():
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = Start()
    app.exec_()
'''

''' This works
import sys
from PyQt5.QtWidgets import QApplication,QWidget
def app():
    #Creating an app
    my_app = QApplication(sys.argv)
    
    #Creating the first widget
    w = QWidget()
    w.setWindowTitle("Testing DrosophilaGP Vizualization GUI")
    w.show()
    #putting the widget in an infinite loop so it doesn't close right away
    sys.exit(my_app.exec_()) #Only exit when you exit from the app
app()
'''

#Source: https://stackoverflow.com/questions/40890632/python-animated-3d-scatterplot-gets-slow
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
import sys

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()

#generate random points from -10 to 10, z-axis positive
pos = np.random.randint(-10,10,size=(1000,3))
pos[:,2] = np.abs(pos[:,2])
print(pos)

sp2 = gl.GLScatterPlotItem(pos=pos)
w.addItem(sp2)

#generate a color opacity gradient
color = np.zeros((pos.shape[0],4), dtype=np.float32)
color[:,0] = 1
color[:,1] = 0
color[:,2] = 0.5
color[0:100,3] = np.arange(0,100)/100.

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    QtGui.QApplication.instance().exec_()