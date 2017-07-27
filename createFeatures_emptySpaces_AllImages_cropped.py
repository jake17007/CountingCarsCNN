from parseXml import numCarsInFile
from parseXml import numSpacesInFile
import os, csv, random, pickle
from PIL import Image
from resize import resize
import numpy as np
import random
from cropPhotos import myCrop

# for each img:
#   turn it to grayscale
#   resize it to a standard size
#   convert it to an array
#   classification <- gettheimageclassification
#   append the img array and classification to the featureset

imgDir = '../PKLot/AllImages/img'
xmlDir = '../PKLot/AllImages/xml'

def getImgArray(img, xml):
    # Open image
    img = myCrop(img, xml)
    #img.show()
    # Resize
    img = img.resize((256,128), Image.ANTIALIAS)
    # Convert image to array
    img = np.asarray(img)
    return img

def getImgClassification(xmlFile):
    numSpaces = numSpacesInFile(xmlFile)
    classVector = []
    for i in range(20):
        if i == numSpaces or (i == 19 and numSpaces > 19):
            classVector.append(1)
        else:
            classVector.append(0)
    return classVector

def createFeaturesAndLabels(rootImgDir, testSize):
    data = []

    i = 1
    for file in os.listdir(rootImgDir):
        img = getImgArray(os.path.join(rootImgDir, file), os.path.join(xmlDir, (os.path.splitext(file)[0] + '.xml')))
        label = getImgClassification(os.path.join(xmlDir, (os.path.splitext(file)[0] + '.xml')))
        data.append([img, label])
        print(i)
        i += 1

    random.shuffle(data)
    data = np.array(data)

    testingSize = int(testSize*len(data))

    train_x = list(data[:,0][:-testingSize])
    train_y = list(data[:,1][:-testingSize])
    test_x = list(data[:,0][-testingSize:])
    test_y = list(data[:,1][-testingSize:])

    print(len(train_x))
    print(len(train_y))
    print(len(test_x))
    print(len(test_y))

    return train_x,train_y,test_x,test_y

train_x,train_y,test_x,test_y = createFeaturesAndLabels(imgDir, 0.1)
# if you want to pickle this data:
with open('feature_set_emptySpaces_AllImages_color_cropped.pickle','wb') as f:
	pickle.dump([train_x,train_y,test_x,test_y],f)
