# python predict.py \
# --model D:\exported_model\coco_val_ssdmobilenet_v2\frozen_inference_graph.pb \
# --labels D:\exported_model\coco_val_ssdmobilenet_v2\mscoco_label_map.pbtxt \
# --image C:\Users\maxua\Desktop \
# --num-classes 1

# import the necessary packages
from object_detection.utils import label_map_util
import tensorflow as tf
import numpy as np
import argparse
import imutils
import cv2
from imutils.paths import list_images
import os

args = {
    "model": "D:\\exported_model\\ring_train_ssd_v2_22k_300620\\frozen_inference_graph.pb",
    "labels": "D:\\exported_model\\ring_train_ssd_v2_5k\\classes.pbtxt",
    "image": "D:\\ring_project\\images_data\\test2",
    # "image": "D:\\ring_project\\images_data\\edit\\hands",
    "num_classes": 1,
    "min_confidence": 0.5
}

# initialize a set of colors for our class labels
COLORS = np.random.uniform(0, 255, size=(args["num_classes"], 3))

# initialize the model
model = tf.Graph()

# create a context manager that makes this model the default one for
# execution
with model.as_default():
    # initialize the graph definition
    graphDef = tf.GraphDef()

    # load the graph from disk
    with tf.gfile.GFile(args["model"], "rb") as f:
        serializedGraph = f.read()
        graphDef.ParseFromString(serializedGraph)
        tf.import_graph_def(graphDef, name="")

# load the class labels from disk
labelMap = label_map_util.load_labelmap(args["labels"])
categories = label_map_util.convert_label_map_to_categories(
    labelMap, max_num_classes=args["num_classes"],
    use_display_name=True)
categoryIdx = label_map_util.create_category_index(categories)

# create a session to perform inference
with model.as_default():
    with tf.Session(graph=model) as sess:
        # grab a reference to the input image tensor and the boxes
        # tensor
        imageTensor = model.get_tensor_by_name("image_tensor:0")
        boxesTensor = model.get_tensor_by_name("detection_boxes:0")

        # for each bounding box we would like to know the score
        # (i.e., probability) and class label
        scoresTensor = model.get_tensor_by_name("detection_scores:0")
        classesTensor = model.get_tensor_by_name("detection_classes:0")
        numDetections = model.get_tensor_by_name("num_detections:0")

        # load the image from disk
        images = list_images(args["image"])
        for image in images:
            path = image
            image = cv2.imread(image)
            (H, W) = image.shape[:2]

            # check to see if we should resize along the width
            if W > H and W > 1000:
                image = imutils.resize(image, width=1000)

            # otherwise, check to see if we should resize along the
            # height
            elif H > W and H > 1000:
                image = imutils.resize(image, height=1000)

            # prepare the image for detection
            (H, W) = image.shape[:2]
            output = image.copy()
            image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
            image = np.expand_dims(image, axis=0)

            # perform inference and compute the bounding boxes,
            # probabilities, and class labels
            (boxes, scores, labels, N) = sess.run(
                [boxesTensor, scoresTensor, classesTensor, numDetections],
                feed_dict={imageTensor: image})

            # squeeze the lists into a single dimension
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            labels = np.squeeze(labels)

            # loop over the bounding box predictions
            for (box, score, label) in zip(boxes, scores, labels):
                # if the predicted probability is less than the minimum
                # confidence, ignore it
                if score < args["min_confidence"]:
                    continue

                # scale the bounding box from the range [0, 1] to [W, H]
                (startY, startX, endY, endX) = box
                startX = int(startX * W)
                startY = int(startY * H)
                endX = int(endX * W)
                endY = int(endY * H)

                # draw the prediction on the output image
                label = categoryIdx[label]
                idx = int(label["id"]) - 1
                # idx = 0
                label = "{}: {:.2f}".format(label["name"], score)
                # label = "{}: {:.2f}".format("person", score)
                cv2.rectangle(output, (startX, startY), (endX, endY),
                            COLORS[idx], 2)
                print(label)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(output, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[idx], 1)

            # show the output image
            cv2.imshow("Output", output)
            name = path.split(os.path.sep)[-1]
            path = path.split(os.path.sep)
            name = name.split(".")[0]
            name = name + "gen"
            name = name + ".jpg"
            path = path[:-1]
            path.append(name)
            path = os.path.sep.join(path)
            # cv2.imwrite(path,output)

            cv2.waitKey(0)