#!/usr/bin/env python
import rospy
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
#from PIL import Image
#import queue
from cv_bridge import CvBridge 
from sensor_msgs.msg import Image as ImageMsg
from std_msgs.msg import UInt8MultiArray as SegMsg # contains ids only, 0-255 range is enough
from std_msgs.msg import MultiArrayLayout as SegMsgLayout
from std_msgs.msg import MultiArrayDimension

CAMERA_TOPIC = "/racer/camera1/image_raw"
RGB_OUTPUT_TOPIC = "/segmentation/rgb"
IDS_OUTPUT_TOPIC = "/segmentation/ids"
computation_rate = 5.0
publish_rate = 30.0


class SegmentationNode:
	#modes = {'custom':3}
	trainId2label={0:[255, 0 ,0], 1:[255, 0, 255], 2: [255, 255, 0], 3:[0,0,255],4:[0,255,0],5:[255,255,255]}
	def __init__(self, model=None, preprocessing=None, device='cpu', classes=3, img_h=376, img_w=672):
		self.classes = classes
		if model==None:
			self.init_model()
		else:
			self.classes = self.model.classes
		self.device = torch.device(device)
		self.model.to(self.device)
		self.model.eval()
		self.preprocessing = preprocessing
		self.bridge = CvBridge()
		self.transf = transforms.ToTensor()
		self.timer = Timer()
		# init subs
		self.sub = rospy.Subscriber(CAMERA_TOPIC, ImageMsg, self.camera_callback, queue_size=20, buff_size=2**24) # buff size in bytes
		# init pubs
		self.rgb_output_publisher = rospy.Publisher(RGB_OUTPUT_TOPIC, ImageMsg, queue_size=20)
		self.ids_output_publisher = rospy.Publisher(IDS_OUTPUT_TOPIC, SegMsg, queue_size=20)
		# init timers
		self.callback_timer = rospy.Timer(rospy.Duration(1/computation_rate), self.compute_output)
		self.callback_timer2 = rospy.Timer(rospy.Duration(1/publish_rate), self.pub_func)
		self.output_image_msg = ImageMsg()
		self.output_ids_msg = SegMsg()
		self.input_msg = ImageMsg()
		self.set_seg_data_layout(img_h, img_w)

	
	def camera_callback(self, msg):
		try:
			self.input_msg = msg
		except:
			raise NotImplementedError
	
	def compute_output(self, event):
		rgb_subscribers_num = self.rgb_output_publisher.get_num_connections()
		ids_subscribers_num = self.ids_output_publisher.get_num_connections()
		if ids_subscribers_num > 0 or ids_subscribers_num > 0:
			#self.timer.start()
			try:
				self.input_cv_image = self.bridge.imgmsg_to_cv2(self.input_msg, "rgb8")
			except:
				pass # empty image msg
			else:
				with torch.no_grad():
					img_tensor = self.transf(self.input_cv_image).unsqueeze(0)
					img_tensor = img_tensor.to(self.device)
					output = self.model(img_tensor).cpu()
					if ids_subscribers_num > 0:
						self.timer.start()
						output_ids = torch.flatten(output).numpy().tolist()
						self.output_ids_msg = SegMsg()
						self.output_ids_msg.data = output_ids
						self.output_ids_msg.layout = self.seg_msg_layout
						self.timer.end()
					if rgb_subscribers_num > 0:
						rgb_output = self.color_map(output.squeeze())
						numpy_image = rgb_output.numpy()
						self.output_image_msg = self.bridge.cv2_to_imgmsg(numpy_image, encoding="rgb8") # set the image encoding(image is displayed in RVIZ according to this encoding)			
			#self.timer.end()
	
	def pub_func(self, event):
		rgb_subscribers_num = self.rgb_output_publisher.get_num_connections()
		ids_subscribers_num = self.ids_output_publisher.get_num_connections()
		self.rgb_output_publisher.publish(self.output_image_msg)
		self.ids_output_publisher.publish(self.output_ids_msg)
	
	def set_seg_data_layout(self, img_h, img_w):
		self.seg_msg_layout = SegMsgLayout()
		mul_array_dims = [MultiArrayDimension(), MultiArrayDimension()]
		mul_array_dims[0].label = "height"
		mul_array_dims[0].size = img_h
		mul_array_dims[0].stride = img_h * img_w
		mul_array_dims[1].label = "width"
		mul_array_dims[1].size = img_w
		mul_array_dims[1].stride = img_w
		self.seg_msg_layout.dim = mul_array_dims

	def init_model(self):
		self.model = test_model(self.classes)
	
	def color_map(self, seg_output):
		# seg_output shape h,w
		r = torch.zeros_like(seg_output, dtype=torch.uint8)
		g = torch.zeros_like(seg_output, dtype=torch.uint8)
		b = torch.zeros_like(seg_output, dtype=torch.uint8)
		for class_id in range(self.classes):
			idx = (seg_output==class_id)
			r[idx] = self.trainId2label[class_id][0]
			g[idx] = self.trainId2label[class_id][1]
			b[idx] = self.trainId2label[class_id][2]
		rgb_img = torch.stack([r,g,b], axis=2) # h,w,c
		return rgb_img

import timeit

class Timer:
	def __init__(self):
		self.start_time = 0
		self.time = 0
		self.it = 0
		self.total_time = 0
	def start(self):
		self.start_time = timeit.default_timer()
	
	def end(self):
		self.it += 1.0
		self.end_time = timeit.default_timer()
		self.time = self.end_time - self.start_time
		self.total_time += self.time
		print('mean time: ', self.total_time/self.it,'s' )



class test_model(nn.Module):
	def __init__(self, classes = 3):
		super(test_model, self).__init__()
		self.layer = nn.Conv2d(3, classes, 1)
		self.get_ids = torch.argmax
		self.classes = classes
	def forward(self, x):
		x = self.layer(x)
		x = self.get_ids(x, dim=1)
		return x.type(torch.uint8)

def main():
	rospy.init_node('semantic_segmentation', anonymous=True)
	node = SegmentationNode()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")


if __name__ == '__main__':
	main() 
		


