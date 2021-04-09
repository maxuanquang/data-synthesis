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
from sort_centroid import Sort_centroid
from imutils.video import FileVideoStream

args = {
	# "model": "D:\\model_zoo\\ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03\\frozen_inference_graph.pb",
	# "model": "D:\\exported_model\\ring_train_ssd_v2_22k_300620\\frozen_inference_graph.pb",
	"model": "D:\\exported_model\\ring_train_ssd_v2_1_5k_130720\\frozen_inference_graph.pb",
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
	# "image": "D:\\ring_project\\images_data\\IMG_4566.MOV",
	# "image": "D:\\ring_project\\images_data\\IMG_4567.MOV",
	# "image": "D:\\ring_project\\images_data\\IMG_4583.MOV",
	# "image": "D:\\ring_project\\images_data\\IMG_4584.MOV",
	"image": "D:\\ring_project\\images_data\\IMG_4590.MOV",
	"image": "D:\\ring_project\\images_data\\IMG_4592.MOV",
	# "image": "D:\\ring_project\\images_data\\IMG_4595.MOV",
	# "image": "D:\\ring_project\\images_data\\IMG_4596.MOV",
	"image": "D:\\ring_project\\images_data\\IMG_4597.MOV",
	"image": "D:\\ring_project\\images_data\\IMG_4602.MOV",
	"image": 0,
	"num_classes": 1,
	"min_confidence": 0.85
}

# initialize the model and create a context manager that makes this model 
# the default one for execution
model = tf.Graph()
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

##########################################################################################################
# # create a session to perform inference
# with model.as_default():
# 	with tf.Session(graph=model) as sess:
# 		#########################
# 		# initialize video writer
# 		#########################
# 		video_capture = cv2.VideoCapture(args["image"])
# 		# video_capture = cv2.VideoCapture(0)
# 		fourcc = cv2.VideoWriter_fourcc(*'XVID')
# 		w = int(video_capture.get(3))
# 		h = int(video_capture.get(4))

# 		out = cv2.VideoWriter('.\\outputs\\ring_hand.avi', fourcc, 30, (w, h))

# 		# grab a reference to the input image tensor and the boxes
# 		# tensor
# 		imageTensor = model.get_tensor_by_name("image_tensor:0")
# 		boxesTensor = model.get_tensor_by_name("detection_boxes:0")

# 		# for each bounding box we would like to know the score
# 		# (i.e., probability) and class label
# 		scoresTensor = model.get_tensor_by_name("detection_scores:0")
# 		classesTensor = model.get_tensor_by_name("detection_classes:0")
# 		numDetections = model.get_tensor_by_name("num_detections:0")
	
# 		while True:
# 			ret, image = video_capture.read()  # frame shape 640*480*3
# 			if ret != True:
# 				break

# 			(H, W) = image.shape[:2]

# 			# check to see if we should resize along the width
# 			if W > H and W > 1000:
# 				image = imutils.resize(image, width=1000)

# 			# otherwise, check to see if we should resize along the
# 			# height
# 			elif H > W and H > 1000:
# 				image = imutils.resize(image, height=1000)

# 			# prepare the image for detection
# 			(H, W) = image.shape[:2]
# 			output = image.copy()
# 			image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
# 			image = np.expand_dims(image, axis=0)

# 			# perform inference and compute the bounding boxes,
# 			# probabilities, and class labels
# 			(boxes, scores, labels, N) = sess.run(
# 				[boxesTensor, scoresTensor, classesTensor, numDetections],
# 				feed_dict={imageTensor: image})

# 			# squeeze the lists into a single dimension
# 			boxes = np.squeeze(boxes)
# 			scores = np.squeeze(scores)
# 			labels = np.squeeze(labels)

# 			rects = []

# 			# loop over the bounding box predictions
# 			for (box, score, label) in zip(boxes, scores, labels):
# 				# if the predicted probability is less than the minimum
# 				# confidence, ignore it
# 				if score < args["min_confidence"]:
# 					continue

# 				# scale the bounding box from the range [0, 1] to [W, H]
# 				(startY, startX, endY, endX) = box
# 				startX = int(startX * W)
# 				startY = int(startY * H)
# 				endX = int(endX * W)
# 				endY = int(endY * H)

# 				rect = [startX,startY,endX,endY,score,label]

# 				rects.append(rect)

# 				# draw the prediction on the output image
# 				label = categoryIdx[label]
# 				# label = {
# 				# 	"id": 1,
# 				# 	"name": "ring"
# 				# }
# 				idx = int(label["id"]) - 1

# 				label = "LabelID: {}; Label name: {}; Confidence score: {:.2f}".format(label["id"], label["name"], score)
# 				# print(label)

# 				cv2.rectangle(output, (startX, startY), (endX, endY),
# 							(0,0,255), 2)
# 				# cv2.circle(output, (int((startX+endX)/2), int((startY+endY)/2)), 2, COLORS[idx], -1)

# 				# y = startY - 10 if startY - 10 > 10 else startY + 10
# 				# cv2.putText(output, label, (startX, y),
# 				#			cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS[idx], 1)

# 			# update our SORT tracker using the computed set of bounding
# 			# box rectangles
# 			rects = np.array(rects)
# 			objects = st.update(rects)

# 			x_safe_max = int(W * 0.95)
# 			x_safe_min = int(W - x_safe_max)
# 			y_safe_max = int(H * 0.9)
# 			y_safe_min = int(H - y_safe_max)

# 			total_In = 0
# 			# loop over the tracked objects to edit
# 			for d in objects:
# 				d = d.astype(np.int32)
# 				centroid = (int((d[0]+d[2])/2),int((d[1]+d[3])/2))
# 				objectID = d[4]
# 				label = d[5]
# 				# check to see if a trackable object exists for the current
# 				# object ID
# 				to = st.trackableObjectsDictionary.get(objectID, None) 

# 				# if there is no existing trackable object, create one
# 				if to is None:
# 					to = TrackableObject(objectID, centroid, label)
# 				else:
# 					to.centroids.append(centroid)

# 				if centroid[0] <= x_safe_min or centroid[0] >= x_safe_max or centroid[1] <= y_safe_min or centroid[1] >= y_safe_max:
# 					to.counted = True  
# 				else:
# 					total_In += 1
# 					to.counted = True

# 				# store the trackable object in our dictionary
# 				st.trackableObjectsDictionary[objectID] = to
# 				# print(len(st.trackableObjectsDictionary.keys()))

# 				# draw both the ID of the object and the centroid of the
# 				# object on the output frame
# 				text = "ID {} Label {}".format(objectID, label)
# 				cv2.putText(output, text, (centroid[0] - 10, centroid[1] - 10),
# 					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
# 				cv2.circle(output, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

# 			# loop over the info tuples and draw them on our frame
# 			total_Rings = 2
# 			# total_see = len(rects)
# 			cv2.putText(output, "Total rings: " + str(total_Rings), (10, H - 60),
# 				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
# 			# cv2.putText(output, "Total rings see: " + str(total_see), (10, H - 80),
# 			# 	cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)			
# 			cv2.putText(output, "Total rings in: " + str(total_In), (10, H - 40),
# 				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# 			# draw a horizontal line in the center of the frame -- once an
# 			# object crosses this line we will determine whether they were
# 			# moving 'up' or 'down'
# 			if total_In == total_Rings:
# 				cv2.rectangle(output, (x_safe_min,y_safe_min), (x_safe_max,y_safe_max), (0, 255, 0), 2)
# 			else:
# 				cv2.rectangle(output, (x_safe_min,y_safe_min), (x_safe_max,y_safe_max), (0, 0, 255), 2)				
# 			# show the output image
# 			cv2.imshow("Output", output)

# 			output = cv2.resize(output,(w,h))
# 			# out.write(output)
# 			if cv2.waitKey(1) & 0xFF == ord('q'):
# 				break
		
# 		video_capture.release()
# 		out.release()
# 		cv2.destroyAllWindows()
# 		print("Save done!")
#########################################################################################################
with model.as_default():
	with tf.Session(graph=model) as sess:
		# initialize video writer
		video_capture = cv2.VideoCapture(args["image"])
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		w = int(video_capture.get(3))
		h = int(video_capture.get(4))
		out = cv2.VideoWriter('.\\outputs\\ring_hand.avi', fourcc, 30, (w, h))

		# grab a reference to the input image tensor and the boxes tensor
		imageTensor = model.get_tensor_by_name("image_tensor:0")
		boxesTensor = model.get_tensor_by_name("detection_boxes:0")
		scoresTensor = model.get_tensor_by_name("detection_scores:0")
		classesTensor = model.get_tensor_by_name("detection_classes:0")
		numDetections = model.get_tensor_by_name("num_detections:0")

		# initialize SORT tracker
		st = Sort_centroid(max_age=100, min_hits=2, cdist_threshold=175)

		# khởi tạo các biến
		total_rings = 0
		missing_frame = 0
		missing_warning = 10

		while True:
			# read frame and resize
			ret, image = video_capture.read()
			if ret != True:
				break
			(H, W) = image.shape[:2]
			if W > H and W > 1000:
				image = imutils.resize(image, width=1000)
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

			# khởi tạo list dùng cho tracking
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

				# thêm vào list rect để feed cho tracking
				rect = [startX,startY,endX,endY,score,label]
				rects.append(rect)

				# print id, score
				label = categoryIdx[label]
				idx = int(label["id"]) - 1
				label = "LabelID: {}; Label name: {}; Confidence score: {:.2f}".format(label["id"], label["name"], score)
				# print(label)

				# draw bounding box
				cv2.rectangle(output, (startX, startY), (endX, endY),
							(0,0,255), 2)

			# update our SORT tracker using the computed set of bounding
			# box rectangles
			rects = np.array(rects)
			objects = st.update(rects)
			left_line = int(0.25*W)
			middle_line = W//2
			right_line = int(W-left_line)
			last_right_line = int(W - left_line // 2)

			# loop over the tracked objects to edit
			for d in objects:
				d = d.astype(np.int32)
				centroid = (int((d[0]+d[2])/2),int((d[1]+d[3])/2))
				objectID = d[4]
				label = d[5]

				# check to see if a trackable object exists for the current
				# object ID
				to = st.trackableObjectsDictionary.get(objectID, None) 
				if to is None:
					to = TrackableObject(objectID, centroid, label)
				else:
					to.centroids.append(centroid)
				# store the trackable object in our dictionary
				st.trackableObjectsDictionary[objectID] = to

				# draw both the ID of the object and the centroid of the
				# object on the output frame
				text = "ID {} Label {}".format(objectID, label)
				cv2.putText(output, text, (centroid[0] - 10, centroid[1] - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
				cv2.circle(output, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

			# hàm theo dõi
			total_see = len(rects)
			ID_del = []
			for objectID in st.trackableObjectsDictionary.keys():
				_object = st.trackableObjectsDictionary[objectID]
				centroids = _object.centroids
				x = [c[0] for c in centroids]
				if len(x) > 1:
					x_near_last = x[-2]
				else:
					x_near_last = x[-1]
				x_last = x[-1]
				direction = x_last - x_near_last

				# add and substract total ring
				# if x_mean < W // 2 and x_last - x_mean > 0:
				if (x_last > right_line and x_last < last_right_line and direction < 0):
					if _object.counted == False:
						total_rings += 1
						_object.counted = True
				if (x_last > right_line and x_last < last_right_line and direction > 0):
					if _object.counted == False:	
						total_rings -= 1
						_object.counted = None
				if x_last <= right_line:
					_object.counted = False	
				if x_last >= last_right_line:
					_object.counted = False	

			# xử lý khi có object mất mà object khác xuất hiện
			# sẽ xóa dictionary của object mất đi, thêm object detect đc vào


			# xử lý khi không có object mất mà object khác xuất hiện


			# xử lý số liệu và in lên màn hình
			if total_rings == total_see:				
				cv2.putText(output, "SAFE", (10, H - 80),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
				missing_frame = 0
			else:
				missing_frame += 1
				if missing_frame < missing_warning:
					cv2.putText(output, "SAFE", (10, H - 80),
						cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
				if missing_frame >= missing_warning:
					cv2.putText(output, "WARNINGS", (10, H - 80),
						cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
			# cv2.line(output, (middle_line,0), (middle_line,H), (0,0,255), 2)
			# cv2.line(output, (left_line,0), (left_line,H), (0,0,255), 2)
			cv2.line(output, (right_line,0), (right_line,H), (0,0,255), 2)
			cv2.line(output, (last_right_line,0), (last_right_line,H), (0,0,255), 2)
			cv2.putText(output, "Total rings: " + str(total_rings), (10, H - 60),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)		
			cv2.putText(output, "Total rings see: " + str(total_see), (10, H - 40),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
			
			# show the output image and record video
			cv2.imshow("Output", output)
			output = cv2.resize(output,(w,h))
			out.write(output)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		
		# ending script
		video_capture.release()
		out.release()
		cv2.destroyAllWindows()
		print("Save done!")