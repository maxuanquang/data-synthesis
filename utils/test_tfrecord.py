# D:\cocosynth\datasets\ring_dataset\records\testing.record
import tensorflow as tf
tf.enable_eager_execution()
from PIL import Image
# import cv2
from matplotlib import pyplot as plt 

tfrecord_dir = ["D:\\cocosynth\\datasets\\ring_dataset\\records\\testing.record"]

def _bytes_feature(value):
	"""Returns a bytes_list from a string / byte."""
	# If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
	if isinstance(value, type(tf.constant(0))):
		value = value.numpy() 
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
	"""Returns a float_list from a float / double."""
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
	"""Returns an int64_list from a bool / enum / int / uint."""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def read_tfrecord(serialized_example):
	feature_description = {
		"image/height": tf.io.FixedLenFeature((), tf.int64),
		"image/width": tf.io.FixedLenFeature((), tf.int64),
		"image/filename": tf.io.FixedLenFeature((), tf.string),
		"image/source_id": tf.io.FixedLenFeature((), tf.string),
		"image/encoded": tf.io.FixedLenFeature((), tf.string),
		"image/format": tf.io.FixedLenFeature((), tf.string),
		"image/object/bbox/xmin": tf.io.FixedLenFeature((), tf.int64),
		"image/object/bbox/xmax": tf.io.FixedLenFeature((), tf.int64),
		"image/object/bbox/ymin": tf.io.FixedLenFeature((), tf.float32),
		"image/object/bbox/ymax": tf.io.FixedLenFeature((), tf.float32),
		"image/object/class/text": tf.io.FixedLenFeature((), tf.string),
		"image/object/class/label": tf.io.FixedLenFeature((), tf.int64),
		"image/object/difficult": tf.io.FixedLenFeature((), tf.int64),
	}
	example = tf.io.parse_single_example(serialized_example, feature_description)
	
	image = tf.io.parse_tensor(example['image/encoded'], out_type = float)
	image_shape = [example['image/height'], example['image/width'], 3]
	image = tf.reshape(image, image_shape)
	
	return image, example['image/object/class/text']

tfrecord_dataset = tf.data.TFRecordDataset(tfrecord_dir)
parsed_dataset = tfrecord_dataset.map(read_tfrecord)
plt.figure(figsize=(10,10))
for i, data in enumerate(parsed_dataset.take(9)):
	img = tf.keras.preprocessing.image.array_to_img(data[0])
	plt.subplot(3,3,i+1)
	plt.imshow(img)
plt.show()
