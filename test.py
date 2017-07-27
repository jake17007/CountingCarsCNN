from parseXml import numCarsInFile
from parseXml import numSpacesInFile
import os
from PIL import Image
import numpy as np
import pickle
from resizeimage import resizeimage
#import tensorflow as tf
import xml.etree.ElementTree as et

'''
imgDir = '../PKLot/AllImages/img'
xmlDir = '../PKLot/AllImages/xml'

def getImgArray(img):
    # Convert to grayscale
    img = Image.open(img)
    # Resize
    img = img.resize((256,128), Image.ANTIALIAS)
    # Convert image to array
    img = np.asarray(img)
    return img

img = getImgArray(os.path.join(imgDir, '0.jpg'))
print (img)
print (img.shape)
'''
'''
def getAllSegmentationPoints(xmlFile):
    coordinates = []
    root = et.parse(xmlFile).getroot()
    for space in root.getchildren():
        for point in space.find('contour').getchildren():
            for coordinate in point.attrib:
                coordinates.append(int(point.attrib[coordinate]))
    return coordinates


firstXmlFile = '../PKLot/PUCPR/xml/0.xml'
secondXmlFile = '../PKLot/PUCPR/xml/1.xml'
firstFile = getAllSegmentationPoints(firstXmlFile)
secondFile = getAllSegmentationPoints(secondXmlFile)

print (firstFile == secondFile)
print ('should be true:',[1,2,3] == [1,2,3])
print ('should be false:',[1,2,3] == [1,2,4])
'''
'''
def getImgClassification():
    numCars = 0
    classVector = []
    for i in range(101):
        if i == numCars:
            classVector.append(1)
        else:
            classVector.append(0)
    return classVector

print(getImgClassification())
'''
'''
data = np.array([1.2, -1.5])
data = np.round(data)
print(data)
'''

predictions = np.asarray([np.arange(101)])
print(predictions.shape)
for i in range(12000):
    predictions = np.concatenate((predictions, np.asarray([np.arange(101)])), axis=0)

predictions = np.argmax(predictions, axis=1)

print(predictions)

np.set_printoptions(threshold=np.inf)
with open('empty_logfile.txt', 'a') as myfile:
    myfile.write(np.array2string(predictions, separator=', '))
    myfile.write('\n\n')
