# 9 July 2013
# Andrea Valente www.create.aau.dk/av
# Lecture 1 Machine Learning lecture notes
#
# 4) see how well the trained perception performs, against new images (test2)

from pylab import imread, mean
import numpy as np
import math

def loadImage(name):
    tmpImg = imread(name)
    return mean(tmpImg,2) # to get a 2-D array

# load generalization set: test2
test2Images = [] # list of couples (image,label)
folder = 'imgs/test2/'
imgNames = [ (folder+'Boss_gs.png',+1),
             (folder+'Brush_gs.png',-1),
             (folder+'Dial_gs.png',-1),
             (folder+'Person_gs.png',+1),
             (folder+'Smile_gs.png',-1)
             ]
for name_and_label in imgNames:
    name,label = name_and_label
    test2Images.append( (loadImage(name) ,label) )

test2flatImages = [] # list of couples (image,label)
for imgAndLabel in test2Images:
    img,label = imgAndLabel
    test2flatImages.append(  (img.ravel(),label) )

# load classifier data (previously created)
f = file("data.bin","rb")
theta = np.load(f)
f.close()

# set up all the right variables to work with the perceptron
fImgs,labels = zip(*test2flatImages)
x = np.array(fImgs)
y = np.array(labels)
n = len(x)

d = 16*16
assert( d == len(theta) )

# Use the perceptron to classify the new images (from test2)

def sign(x):
    ''' x is a scalar '''
    return math.copysign(1, x)

# linear classifiers through origin
def f(x,theta):
    ''' x is a numpy array of length d,
        theta also. '''
    return sign( np.dot(theta.transpose(), x) )

def trainingError(x,y,theta):
    ''' x is a numpy array of length n (of images of length d),
        y is an array of labels (in {-1,1}) of length d,
        theta is a numpy array of length d. '''
    n = len(x)
    total = 0.0
    for t in range(n):
        if not (f(x[t],theta) == y[t]):
            total += 1  # add all errors
    return 1.0/n * total

print "\nTesting the performance of the classifier.\n"
for t in range(n):
    print imgNames[t]
    if f(x[t],theta)>0:
        print "image accepted"
    else:
        print "rejected"

e = trainingError(x,y,theta)
print "average classification error = ",e

