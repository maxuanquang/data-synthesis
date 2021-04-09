# import the necessary packages
from object_detection.utils import label_map_util
import tensorflow as tf
import numpy as np
import argparse
import imutils
import cv2
from imutils.paths import list_images
from pyimagesearch.utils.trackableobject import TrackableObject
from sort import Sort
from imutils.video import FileVideoStream

args = {
	# "model": "D:\\model_zoo\\ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03\\frozen_inference_graph.pb",
	"model": "D:\\exported_model\\ring_train_ssd_v2_22k_300620\\frozen_inference_graph.pb",
	"labels": "D:\\exported_model\\ring_train_ssd_v2_5k\\classes.pbtxt",
	# "image": "D:\\ring_project\\images_data\\IMG_4394.MOV",
	# "image": "D:\\ring_project\\images_data\\IMG_4405.MOV",
	# "image": "D:\\ring_project\\images_data\\IMG_4395.mp4",
	# "image": "D:\\ring_project\\images_data\\test2\\IMG_4292.MOV",
	# "image": "D:\\ring_project\\images_data\\IMG_4486.MOV",
	# "image": "D:\\ring_project\\images_data\\IMG_4487.MOV",
	# "image": "D:\\ring_project\\images_data\\IMG_4488.MOV",
	# "image": "D:\\ring_project\\images_data\\IMG_4489.MOV",
	# "image": "D:\\ring_project\\images_data\\IMG_4505.MOV",
	# "image": "D:\\ring_project\\images_data\\IMG_4506.MOV",
	"image": "D:\\ring_project\\images_data\\IMG_4566.MOV",
	# "image": "D:\\ring_project\\images_data\\IMG_4567.MOV",
	"num_classes": 1,
	"min_confidence": 0.72
}

################################################################
# initialize SORT tracker
################################################################
total_Rings = 0
total_In = 0
total_Out = 0

st = Sort(max_age=100, min_hits=0, iou_threshold=0.01)

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
		#########################
		# initialize video writer
		#########################
		video_capture = cv2.VideoCapture(args["image"])
		# video_capture = cv2.VideoCapture(0)
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		w = int(video_capture.get(3))
		h = int(video_capture.get(4))

		out = cv2.VideoWriter('.\\outputs\\ring_hand.avi', fourcc, 30, (w, h))

		# grab a reference to the input image tensor and the boxes
		# tensor
		imageTensor = model.get_tensor_by_name("image_tensor:0")
		boxesTensor = model.get_tensor_by_name("detection_boxes:0")

		# for each bounding box we would like to know the score
		# (i.e., probability) and class label
		scoresTensor = model.get_tensor_by_name("detection_scores:0")
		classesTensor = model.get_tensor_by_name("detection_classes:0")
		numDetections = model.get_tensor_by_name("num_detections:0")
	
		while True:
			ret, image = video_capture.read()  # frame shape 640*480*3
			if ret != True:
				break

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

			rects = []

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

				rect = [startX,startY,endX,endY,score,label]

				rects.append(rect)

				# draw the prediction on the output image
				label = categoryIdx[label]
				# label = {
				# 	"id": 1,
				# 	"name": "ring"
				# }
				idx = int(label["id"]) - 1

				label = "LabelID: {}; Label name: {}; Confidence score: {:.2f}".format(label["id"], label["name"], score)
				print(label)

				cv2.rectangle(output, (startX, startY), (endX, endY),
							(0,0,255), 2)
				# cv2.circle(output, (int((startX+endX)/2), int((startY+endY)/2)), 2, COLORS[idx], -1)

				# y = startY - 10 if startY - 10 > 10 else startY + 10
				# cv2.putText(output, label, (startX, y),
				#			cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS[idx], 1)

			# update our SORT tracker using the computed set of bounding
			# box rectangles
			rects = np.array(rects)
			objects = st.update(rects)

			x_safe_max = int(W * 0.95)
			x_safe_min = int(W - x_safe_max)
			y_safe_max = int(H * 0.9)
			y_safe_min = int(H - y_safe_max)

			total_In = 0
			# loop over the tracked objects to edit
			for d in objects:
				d = d.astype(np.int32)
				centroid = (int((d[0]+d[2])/2),int((d[1]+d[3])/2))
				objectID = d[4]
				label = d[5]
				# check to see if a trackable object exists for the current
				# object ID
				to = st.trackableObjectsDictionary.get(objectID, None) 

				# if there is no existing trackable object, create one
				if to is None:
					to = TrackableObject(objectID, centroid, label)

				# otherwise, there is a trackable object so we can utilize it
				# to determine direction
				else:
					# the difference between the y-coordinate of the *current*
					# centroid and the mean of *previous* centroids will tell
					# us in which direction the object is moving (negative for
					# 'up' and positive for 'down')
					to.centroids.append(centroid)

				if centroid[0] <= x_safe_min or centroid[0] >= x_safe_max or centroid[1] <= y_safe_min or centroid[1] >= y_safe_max:
					total_Out += 1
					to.counted = True  
				else:
					total_In += 1
					to.counted = True

				# store the trackable object in our dictionary
				st.trackableObjectsDictionary[objectID] = to
				print(len(st.trackableObjectsDictionary.keys()))

				# draw both the ID of the object and the centroid of the
				# object on the output frame
				text = "ID {} Label {}".format(objectID, label)
				cv2.putText(output, text, (centroid[0] - 10, centroid[1] - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
				cv2.circle(output, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

			# loop over the info tuples and draw them on our frame
			total_Rings = 4
			# total_see = len(rects)
			cv2.putText(output, "Total rings: " + str(total_Rings), (10, H - 60),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
			# cv2.putText(output, "Total rings see: " + str(total_see), (10, H - 80),
			# 	cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)			
			cv2.putText(output, "Total rings in: " + str(total_In), (10, H - 40),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

			# draw a horizontal line in the center of the frame -- once an
			# object crosses this line we will determine whether they were
			# moving 'up' or 'down'
			if total_In == total_Rings:
				cv2.rectangle(output, (x_safe_min,y_safe_min), (x_safe_max,y_safe_max), (0, 255, 0), 2)
			else:
				cv2.rectangle(output, (x_safe_min,y_safe_min), (x_safe_max,y_safe_max), (0, 0, 255), 2)				
			# show the output image
			cv2.imshow("Output", output)

			output = cv2.resize(output,(w,h))
            # out.write(output)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		
		video_capture.release()
		out.release()
		cv2.destroyAllWindows()
		print("Save done!")