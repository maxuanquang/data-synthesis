import cv2
from imutils.paths import list_images
from PIL import Image
import os

folder_path = "D:\\ring_project\\images_data\\photoshop_goldring"
image_paths = list(list_images(folder_path))
for image_path in image_paths:
    image = Image.open(image_path)

    new_name = image_path.split(os.path.sep)[-1].split(".")
    new_name[0] = new_name[0] + "_edit_20"
    new_name = new_name[0] + "." + new_name[1]
    new_path = list(image_path.split(os.path.sep)[:-1])
    new_path.append(new_name)
    new_path = os.path.sep.join(new_path)

    size = image.size
    new_size = (int(size[0] * 0.2), int(size[1] * 0.2))
    image = image.resize(new_size)
    image.save(new_path)
    image.close()
