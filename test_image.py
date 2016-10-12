
import numpy as np
import pylab
import gzip as gz
from PIL import Image, ImageOps
import numpy
import glob
import pickle
from numpy.random import permutation



# open random image of dimensions 639x516




# dimensions are (height, width, channel)
en = 1280-224
step = 224 * 0.5
image_list = []
label = []

for filename in glob.glob('/Users/Mint/Documents/2_2558/Deep neural network/Shortfile_Pic_C1/Fig_ES/*.jpg'): #assuming gif

    #filename = tf.train.string_input_producer([filename])
    #reader = tf.WholeFileReader()
    #key, value = reader.read(filename)
    #example = tf.image.decode_jpeg(value)
    #example = tf.image.rgb_to_grayscale(example)
    #image_list = image_list.append(example)

    #label.append(1)
    im = Image.open(filename).convert('L')
    label.append(0)
    imcur = im
    image_list.append(np.array(imcur.getdata())/255)

    #size = (300, 300)
    #im = ImageOps.fit(im, size, Image.ANTIALIAS)
    #img = np.array(im.getdata())/255
    #image_list.append(img)


for filename in glob.glob('/Users/Mint/Documents/2_2558/Deep neural network/Shortfile_Pic_C1/Fig_NR/*.jpg'): #assuming gif

    #label.append(2)
    #im = Image.open(filename).convert('L')
    #size = (300, 300)
    #im = ImageOps.fit(im, size, Image.ANTIALIAS)
    #img = np.array(im.getdata())/255
    #print(np.shape(img))
    #image_list.append(img)
    im = Image.open(filename).convert('L')
    label.append(1)
    imcur = im
    image_list.append(np.array(imcur.getdata())/255)


for filename in glob.glob('/Users/Mint/Documents/2_2558/Deep neural network/Shortfile_Pic_C1/Fig_AF/*.jpg'): #assuming gif

    #label.append(2)
    #im = Image.open(filename).convert('L')
    #size = (300, 300)
    #im = ImageOps.fit(im, size, Image.ANTIALIAS)
    #img = np.array(im.getdata())/255
    #print(np.shape(img))
    #image_list.append(img)
    im = Image.open(filename).convert('L')
    label.append(2)
    imcur = im
    image_list.append(np.array(imcur.getdata())/255)


imarray = np.array(image_list)
imarray = imarray.astype(numpy.float64)
label = np.array(label)
label = label.astype(numpy.float64)
perm = permutation(len(label))
imarray = imarray[perm]
label = label[perm]
print(perm)

cut = 2841
s=4059
print(np.shape(imarray))
train = imarray[0:cut:1]
ltrain = label[0:cut:1]

test = imarray[cut:s:1]
ltest = label[cut:s:1]


print(imarray.dtype)
print(label.dtype)
print(ltest)
