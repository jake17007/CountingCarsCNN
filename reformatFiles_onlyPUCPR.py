import os
from shutil import copyfile

# Put all the files into one folder
# Rename each img file to a unique number with a matching xml file

def moveFiles(rootDir, newImgDir, newXmlDir):
    # for all files in root:
    #   if file is jpg:
    #       copy it to the new img directory
    #   if file is xml:
    #       copy it to the new xml directory
    numImg = 0
    numXml = 0

    curImg = ''
    curXml = ''
    shouldBeJpg = True

    newFilename = 0

    for root, dirs, files in os.walk(rootDir):
        for file in files:
            path = os.path.join(root, file)
            # The XML file for '2012-11-06_18_48_46.jpg' is missing
            if file.endswith('.jpg') and file != '2012-11-06_18_48_46.jpg':
                if shouldBeJpg == False:
                    print('something went wrong ' + path)
                curImg = os.path.splitext(file)[0]
                copyfile(path, os.path.join(newImgDir, str(newFilename) + '.jpg'))
                numImg += 1
                shouldBeJpg = False
            if file.endswith('.xml'):
                curXml = os.path.splitext(file)[0]
                copyfile(path, os.path.join(newXmlDir, str(newFilename) + '.xml'))
                numXml += 1
                shouldBeJpg = True
                newFilename += 1


    print('number of image files = ' + str(numImg))
    print('number of xml files = ' + str(numXml))

moveFiles('../PKLot/PKLot/PUCPR', '../PKLot/PUCPR/img', '../PKLot/PUCPR/xml')
