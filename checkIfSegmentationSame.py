import xml.etree.ElementTree as et
import os

xmlDir = '../PKLot/PUCPR/xml'

def getAllSegmentationPoints(xmlFile):
    coordinates = []
    root = et.parse(xmlFile).getroot()
    for space in root.getchildren():
        for point in space.find('contour').getchildren():
            for coordinate in point.attrib:
                coordinates.append(int(point.attrib[coordinate]))
    return coordinates

for fileA in os.listdir(xmlDir):
    fileAPoints = getAllSegmentationPoints(os.path.join(xmlDir, fileA))
    for fileB in os.listdir(xmlDir):
        if fileA != fileB:
            fileBPoints = getAllSegmentationPoints(os.path.join(xmlDir, fileB))
            if fileAPoints != fileBPoints:
                print('These are not the same! : ' + fileA + ' != ' + fileB)
    print(fileA)
