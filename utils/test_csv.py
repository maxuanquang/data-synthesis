# import the necessary packages
from bs4 import BeautifulSoup
from PIL import Image
import tensorflow as tf
import os
import cv2

def main():
    # initialize a data dictionary used to map each image filename
    # to all bounding boxes associated with the image, then load
    # the contents of the annotations file
    D = {}
    rows = open("D:\\cocosynth\\datasets\\ring_dataset\\output\\csv\\ring.csv").read().strip().split("\n") # strip() bỏ kí tự space đầu-cuối line
    count = 0

    # loop over the individual rows, skipping the header
    for row in rows[1:]:
        count += 1
        print(count)

        # break the row into components
        row = row.split(",")
        (imagePath, label, startX, startY, endX, endY) = row
        (startX, startY) = (float(startX), float(startY))
        (endX, endY) = (float(endX), float(endY))

        # build the path to the input image, then grab any other
        # bounding boxes + labels associated with the image
        # path, labels, and bounding box lists, respectively
        p = imagePath
        b = D.get(p, [])

        # build a tuple consisting of the label and bounding box,
        # then update the list and store it in the dictionary
        b.append((label, (startX, startY, endX, endY)))
        D[p] = b # D = {"imageA.jpg":[(pedestrian,(1,2,3,4)), (pedestrian,(1,2,3,4)) ]}
    
    for path in D.keys():
        image = cv2.imread(path)
        # for (label,(xmin,ymin,xmax,ymax)) in D[path]:
        for content in D[path]:
            xmin = int(content[1][0])
            ymin = int(content[1][1])
            xmax = int(content[1][2])
            ymax = int(content[1][3])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
        cv2.imshow("quang",image)
        cv2.waitKey(0)
            

main()