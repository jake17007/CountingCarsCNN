from PIL import Image
import xml.etree.ElementTree as et
import os
#from drawBoxes import drawBoxes

def getAllSegmentationPoints(xmlFile):
    coordinates = []
    root = et.parse(xmlFile).getroot()
    for space in root.getchildren():
        for point in space.find('contour').getchildren():
            for coordinate in point.attrib:
                coordinates.append(int(point.attrib[coordinate]))
    return coordinates
'''
for fileA in os.listdir(xmlDir):
    fileAPoints = getAllSegmentationPoints(os.path.join(xmlDir, fileA))
    for fileB in os.listdir(xmlDir):
        if fileA != fileB:
            fileBPoints = getAllSegmentationPoints(os.path.join(xmlDir, fileB))
            if fileAPoints != fileBPoints:
                print('These are not the same! : ' + fileA + ' != ' + fileB)
    print(fileA)
'''
def getBounds(xml):
    leftBound = 1280/2
    rightBound = 1280/2
    upperBound = 720/2
    lowerBound = 720/2
    root = et.parse(xml).getroot()
    for space in root.getchildren():
        for point in space.find('contour').getchildren():
            x = float(point.attrib['x'])
            y = float(point.attrib['y'])
            if x < leftBound:
                leftBound = x
            if x > rightBound:
                rightBound = x
            if y < upperBound:
                upperBound = y
            if y > lowerBound:
                lowerBound = y
    #print(leftBound, upperBound, rightBound, lowerBound)
    return leftBound, upperBound, rightBound, lowerBound
    '''
    upperLeft = (leftBound, upperBound)
    upperRight = (rightBound, upperBound)
    lowerLeft = (leftBound, lowerBound)
    lowerRight = (rightBound, lowerBound)
    '''

def myCrop(img, xml):
    img = Image.open(img)
    leftBound, upperBound, rightBound, lowerBound = getBounds(xml)
    newImg = img.crop((leftBound, upperBound, rightBound, lowerBound))
    #newImg.show()
    return newImg



myCrop('../PKLot/AllImages/img/5926.jpg', '../PKLot/AllImages/xml/5926.xml')
