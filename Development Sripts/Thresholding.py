import matplotlib.pyplot as plt
import numpy as np
import math
import csv
import sys

#np.set_printoptions(threshold=sys.maxsize) #Print out full arrays/data

'''
Reads and stores the intensity matrix for one z-slice 
Cuts off first row and first column
'''
def GetSliceIntensities(path):
	with open(path, 'r') as csv_file:
		matrix = []
		csv_reader = csv.reader (csv_file)
		for row in csv_reader:
			matrix.append(row)
		I = np.array(matrix)
		I = I[1:,1:]
	return I

'''
Creates, displays and closes a 2D plot 
'''
def TwoDimPlot(x,y):
	plt.plot(x,y)
	plt.show()
	plt.close()

'''
L - number of possible intensity values
I - matrix of intensity values 
Counts number of pixels of each intensity value
'''
def hist(L, I):
	h = np.zeros(L) #1D array that stores the number of pixels with each intensity value
	size = I.shape
	for i in range (0, size[0]): #For every row
		for j in range (0, size[1]): #For each column value in the row 
			index = int(I[i][j])
			h[index] = h[index] + 1
	return h

'''
h - original histogram values 
n - number of pixels 
'''	
def normHist(h,n,L):
	nh = np.zeros(L)
	for i in range (0, len(h)):#For every element in h
		nh[i] = h[i]/n
	return nh

def eqHist(L, nh,I):
	
	#Cumulative sum 
	c = np.zeros(L)
	c[0] = nh[0]
	for i in range(1, L):
		c[i] = c[i-1]+ nh[i]
	
	#Transforming original image intensities to improve contrast
	size = I.shape	
	for i in range (0, size[0]): #For every row
		for j in range (0, size[1]): #For each column value in the row
		    newI = math.floor((L-1)* c[int(I[i][j])])
		    I[i][j] = newI
	return I

'''
Dividing the total data range (maximum-minimum) pixel value into 256 separate bins with equal widths. 
These bins are then used to sort pixels that fall within a certain range into the appropriate bin. 
(Source: https://petebankhead.gitbooks.io/imagej-intro/content/chapters/thresholding/thresholding.html)
'''
#def histBins():
	#return 
		
#Main
I = GetSliceIntensities('Input/Slice1.csv') #I is the matrix of intensities
n = I.shape[0]*I.shape[1] #n is the number of pixels in the z-slice 

#Note:To access an element in I use I[row][col]; The first element is I[0][0]

#GRAY-LEVEL HISTOGRAM
#Note: The z-slices have 16 bit resolution. The grayscale intensity values range from 0-65535
L = 65535
h = hist(L, I)
print(h)
bins = np.arange(0,L,1)
TwoDimPlot(bins,h)

#NORMALIZED HISTOGRAM (or) Probability density function
nh = normHist(h,n,L)
print(nh)
TwoDimPlot(bins,nh)

#EQUALIZED HISTOGRAM
equalizedI = eqHist(L, nh, I)
eh = hist(L, equalizedI)
print(eh)
TwoDimPlot(bins,eh)

#Checking if binning is useful
plt.hist(I, bins = 256)
plt.title("Histogram with 256 bins")
plt.show()

#Otsu's algorithm - Find a threshold t that minimizes the within class variance (weighted sum of )
#Compute histogram 
