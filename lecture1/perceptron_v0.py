# 9 July 2013
# Andrea Valente www.create.aau.dk/av
# Lecture 1 Machine Learning lecture notes
#
# 1) load all 7 images, as 16x16 grayscale images
# 2) Alien, Bee, Female and Male should be accepted
# 3) train a perceptron to recognize the "good" images and reject the bad ones
# 4) see how well the trained perception performs, against new images (test2)

from pylab import imread, imshow, gray, mean
a = imread('imgs/positive/alien_gs.png')
#generates a RGB image, so do
aa=mean(a,2) # to get a 2-D array

for i in range(16):
    for j in range(16):
        print "%3d" % int(aa[i][j]*255) , 
    print


