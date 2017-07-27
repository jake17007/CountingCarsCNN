from PIL import Image
def resize(img, box, fit, out):
    '''Downsample the image.
    @param img: Image -  an Image-object
    @param box: tuple(x, y) - the bounding box of the result image
    @param fit: boolean - crop the image to fill the box
    @param out: file-like-object - save the image into the output stream
    '''
    #preresize image with factor 2, 4, 8 and fast algorithm
    factor = 1
    while img.size[0]/factor > 2*box[0] and img.size[1]*2/factor > 2*box[1]:
        factor *=2
    if factor > 1:
        img.thumbnail((img.size[0]/factor, img.size[1]/factor), Image.NEAREST)

    #calculate the cropping box and get the cropped part
    if fit:
        x1 = y1 = 0
        x2, y2 = img.size
        wRatio = 1.0 * x2/box[0]
        hRatio = 1.0 * y2/box[1]
        if hRatio > wRatio:
         y1 = int(y2/2-box[1]*wRatio/2)
         y2 = int(y2/2+box[1]*wRatio/2)
        else:
         x1 = int(x2/2-box[0]*hRatio/2)
         x2 = int(x2/2+box[0]*hRatio/2)
        img = img.crop((x1,y1,x2,y2))

    #Resize the image with best quality algorithm ANTI-ALIAS
    img.thumbnail(box, Image.ANTIALIAS)

    #save it into a file-like object
    if out:
        img.save(out, "JPEG", quality=75)
    
    return img

# https://www.united-coders.com/christian-harms/image-resizing-tips-every-coder-should-know/
#resize
