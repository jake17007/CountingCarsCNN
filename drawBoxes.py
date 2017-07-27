from PIL import Image, ImageDraw
import xml.etree.ElementTree as et

imgFile = '../PKLot/AllImages/img/0.jpg'
xmlFile = '../PKLot/AllImages/xml/0.xml'

'''
<point x="278" y="230" />
<point x="290" y="186" />
<point x="324" y="185" />
<point x="308" y="230" />
'''
def drawBoxes(imgFile, xmlFile):
    img = Image.open(imgFile)
    draw = ImageDraw.Draw(img)

    root = et.parse(xmlFile).getroot()
    for space in root.getchildren():
        x = []
        y = []
        for point in space[1].getchildren():
            x.append(int(point.attrib['x']))
            y.append(int(point.attrib['y']))
        draw.line((x[0],y[0], x[1],y[1]), fill=(255,255,0), width=3)
        draw.line((x[1],y[1], x[2],y[2]), fill=(255,255,0), width=3)
        draw.line((x[2],y[2], x[3],y[3]), fill=(255,255,0), width=3)
        draw.line((x[3],y[3], x[0],y[0]), fill=(255,255,0), width=3)
    img.show(255)

drawBoxes(imgFile, xmlFile)
