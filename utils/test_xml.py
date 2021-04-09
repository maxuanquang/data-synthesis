# import the necessary packages
from config import config
from pyimagesearch.utils.tfannotation import TFAnnotation
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf
import os
from imutils.paths import list_files
from imutils.paths import list_images
from bs4 import BeautifulSoup
import cv2

xml_paths = "D:\\cocosynth\\datasets\\ring_dataset\\output\\xml"
image_paths = "D:\\cocosynth\\datasets\\ring_dataset\\output\\images"

# xml_paths = "D:\\ring_project\\images_data\\edit\\annotations_hand"
# image_paths = "D:\\ring_project\\images_data\\edit\\photoshop_hand"

xml_paths = list(list_files(xml_paths))
image_paths = list(list_images(image_paths))

for xml_path in xml_paths:
    contents = open(xml_path).read() # đọc toàn bộ file
    soup = BeautifulSoup(contents,"xml")
    image_path = soup.find("path")
    # print(image_path)
    image = cv2.imread(image_path.text)
    object_ = soup.find_all("object")
    for obj in object_:
        bbox = soup.find_all("bndbox")
        for bounding_box in bbox:
            xmin = int(bounding_box.find("xmin").text)
            ymin = int(bounding_box.find("ymin").text)
            xmax = int(bounding_box.find("xmax").text)
            ymax = int(bounding_box.find("ymax").text)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
    cv2.imshow("qunag",image)
    cv2.waitKey(0)