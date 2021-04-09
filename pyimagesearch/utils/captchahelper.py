# import the neccessary packages
import imutils
import cv2

def preprocess(image, desire_width, desire_height):
    # grab the dimensions of the images, then initialize
    # the padding values
    (h,w)=image.shape[:2]

    # if the width is greater than the height then resize along
    # the width
    if w>h:
        image=imutils.resize(image,width=desire_width)
    
    # contrast'
    else:
        image=imutils.resize(image,height=desire_height)
    
    # determine the padding values for the width and height to
    # obtain the target dimensions
    padW = int((desire_width - image.shape[1]) / 2.0)
    padH = int((desire_height - image.shape[0]) / 2.0)

    # pad the image then apply one more resizing to handle any
    # rounding issues
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
    cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (desire_width, desire_height))

    # return the pre-processed image
    return image