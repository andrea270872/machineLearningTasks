# 9 July 2013
# Andrea Valente www.create.aau.dk/av
# Lecture 1 Machine Learning lecture notes
#
# 1) load all 7 images, as 16x16 grayscale images
# 2) Alien, Bee, Female and Male should be accepted
# 3) train a perceptron to recognize the "good" images and reject the bad ones
# 4) see how well the trained perception performs, against new images (test2)

from pylab import imread, mean
import numpy as np
import math

def loadImage(name):
    tmpImg = imread(name)
    return mean(tmpImg,2) # to get a 2-D array

def printImage(img):
    for i in range(16):
        for j in range(16):
            print "%3d" % int(img[i][j]*255) , 
        print

def printFlattenedImage(img):
    for i in range(16):
        for j in range(16):
            print "%3d" % int(img[i*16+j]*255) ,
        print

images = [] # list of couples (image,label)

# 1a) load all positive images
folder = 'imgs/positive/'
imgNames = [ folder+'Alien_gs.png',
             folder+'Bee_gs.png',
             folder+'Female_gs.png',
             folder+'Male_gs.png' ]

for name in imgNames:
    images.append( (loadImage(name) ,+1) )

# 1b) load all negative images
folder = 'imgs/negative/'
imgNames = [ folder+'Anchor_gs.png',
             folder+'Blue key_gs.png',
             folder+'Danger_gs.png' ]
for name in imgNames:
    images.append( (loadImage(name) ,-1) )


### Print all images in text-mode
##for imgAndLabel in images:
##    img,label = imgAndLabel
##    printImage( img )
##    print

# Before the peceptron can work, images should be converted into
# a single array of 16*16 = 256 values -> use ravel()

flatImages = [] # list of couples (image,label)
for imgAndLabel in images:
    img,label = imgAndLabel
    flatImages.append(  (img.ravel(),label) )

### Print all flattened images in text-mode
##for imgAndLabel in flatImages:
##    img,label = imgAndLabel
##    printFlattenedImage( img )
##    print

# 2) set up all the right variables to work with the perceptron
fImgs,labels = zip(*flatImages)
x = np.array(fImgs)
y = np.array(labels)
n = len(x)
d = 16*16

# 3) set up a perceptron and train it
# A perceptron should just be an array theta of weights, length 16*16 = 256
theta = np.zeros(d)

### debug
##print x, len(x)
##print y, len(y)

def sign(x):
    ''' x is a scalar '''
    return math.copysign(1, x)

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

##for t in range(n):
##    print f(x[t],theta), " =?= " , y[t]

e = trainingError(x,y,theta)
print "training error = ",e

# 3b) iterate to minimize the error
# TODO


