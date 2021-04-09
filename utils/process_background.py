from PIL import Image
from imutils.paths import list_images
import os

paths = "D:\\cocosynth\\datasets\\ring_dataset\\plain_background"
imagePaths = list_images(paths)

for imagePath in imagePaths:
    background = Image.open(imagePath)
    w,h = background.size
    if w < 1280 or h < 720:
        background.close()
        os.remove(imagePath)
