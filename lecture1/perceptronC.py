# 9 July 2013
# Andrea Valente www.create.aau.dk/av
# Lecture 1 Machine Learning lecture notes
#
# Double checking that the perceptron learned correctly!

from pylab import imread, mean
import numpy as np
import math

def loadImage(name):
    tmpImg = imread(name)
    return mean(tmpImg,2) # to get a 2-D array

# load generalization set: test2
folder1 = 'imgs/positive/'
folder2 = 'imgs/negative/'
imgNames = [ (folder1+'Alien_gs.png', +1),
             (folder1+'Bee_gs.png',   +1),
             (folder1+'Female_gs.png',+1),
             (folder1+'Male_gs.png',  +1),
             
             (folder2+'Anchor_gs.png',  -1),
             (folder2+'Blue key_gs.png',-1),
             (folder2+'Danger_gs.png',  -1) ]

images = [] # list of couples (image,label)
for name_and_label in imgNames:
    name,label = name_and_label
    images.append( (loadImage(name) ,label) )

flatImages = [] # list of couples (image,label)
for imgAndLabel in images:
    img,label = imgAndLabel
    flatImages.append(  (img.ravel(),label) )


# To test the robustness of the classifier,
# here I add random noise to the images:

## this is uniform noise, perhaps try only changing few pixels
def noise(k,n):
    return np.random.uniform(-k,k, size=n)
    #return np.zeros(n) # for perfect images, without noise
        

for i in range(len(flatImages)):
    img,label = flatImages[i]
    flatImages[i] = ( img + noise(10,len(img)) ,label)



# load classifier data (previously created)
f = file("data.bin","rb")
theta = np.load(f)
f.close()

# set up all the right variables to work with the perceptron
fImgs,labels = zip(*flatImages)
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

