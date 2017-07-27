import xml.etree.ElementTree as et

def numCarsInFile(file):
    root = et.parse(file).getroot()
    numCars = 0
    for space in root.getchildren():
        if 'occupied' in space.attrib and space.attrib['occupied'] != '0':
                numCars += 1
    return numCars

def numSpacesInFile(file):
    root = et.parse(file).getroot()
    numSpaces = 0
    for space in root.getchildren():
        if 'occupied' in space.attrib and space.attrib['occupied'] == '0':
                numSpaces += 1
    return numSpaces

# For testing
#file = '../PKLot/AllImages/xml/2012-09-11_15_16_58.xml'
#print (numCarsInFile(file))
