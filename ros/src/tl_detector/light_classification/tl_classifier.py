from styx_msgs.msg import TrafficLight

import numpy as np
import os
import sys
import tensorflow as tf
import time

from collections import defaultdict

#TODO: make model a shared resource
faster_rcnn_sim_model = '/home/grigory/classes/SDC/CarND-Capstone/training_classifier/models/faster_rcnn_local/frozen_inference_graph.pb'

CLASS_TO_TRAFFIC_LIGHT = {
    0: TrafficLight.RED,
    1: TrafficLight.YELLOW,
    2: TrafficLight.GREEN,
    4: TrafficLight.UNKNOWN
}

class TLClassifier(object):
    def __init__(self):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(faster_rcnn_sim_model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.sess = tf.Session(graph=self.detection_graph)

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def __del__(self):
        self.sess.close()
        self.detection_graph.close()
    
    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image_np = self.load_image_into_numpy_array(image)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        (boxes, scores, classes, num) = self.sess.run(
              [self.detection_boxes, self.detection_scores,
               self.detection_classes, self.num_detections],
              feed_dict={self.image_tensor: image_np_expanded})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        min_score_thresh = .50

        light = TrafficLight.UNKNOWN

        for i in range(boxes.shape[0]):
            if scores[i] > min_score_thresh:
                    c_l = CLASS_TO_TRAFFIC_LIGHT[classes[i]]
                    if c_l < light:
                        light = c_l

        return light
