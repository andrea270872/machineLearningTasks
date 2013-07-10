# 9 July 2013
# Andrea Valente www.create.aau.dk/av
# Lecture 1 Machine Learning lecture notes
#
# 1) load all 7 images, as 16x16 grayscale images
# 2) Alien, Bee, Female and Male should be accepted
# 3) train a perceptron to recognize the "good" images and reject the bad ones
# 4) see how well the trained perception performs, against new images (test2)

from pylab import imread, mean

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

# Print all flattened images in text-mode
for imgAndLabel in flatImages:
    img,label = imgAndLabel
    printFlattenedImage( img )
    print
    
# 3) set up a perceptron and train it
# A perceptron should just be an array theta of weights, length 16*16 = 256
theta = zeros(16*16)

# 3b) iterate to minimize the error
# TODO



