import imutils

def image_pyramid(image,scale = 1.5,minSize = (224,224)):
    """
    Parameters
    ----------
    image : input image.
    scale : resizing the scale of the image. The default is 1.5.
    minSize : Controls the minimum size of an output image. The default is (224,224).

    Returns
    -------
    None.

    """
    yield image
    
    while True:
        
        w = int(image.shape[1] / scale)
        image = imutils.resize(image,width = w)
        
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        
        yield image