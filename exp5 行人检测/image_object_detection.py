# -*- coding: utf-8 -*-
import numpy as np
import os
import tensorflow as tf
import time
import cv2
import sys

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

#if tf.__version__ < '1.4.0':
#    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')
#添加python环境变量
CWD_PATH = os.getcwd()
sys.path.append(CWD_PATH)
SLIM_PATH = os.path.join(CWD_PATH,'slim')
sys.path.append(SLIM_PATH)

#已经训练好的模型路径
MODEL_NAME = 'pedestrian_detection'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join(CWD_PATH,'annotations\label_map.pbtxt')

NUM_CLASSES = 1

#将模型载入内存
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
		
#载入label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#测试图片路径
image_path = 'test_image.jpg'

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:          
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        
        image_np = cv2.imread(image_path)
        
        
        start = time.clock()
        
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # 开始检测
        (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
        
        end = time.clock()
        
        # 可视化检测结果
        vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
        print('test time: ','\n', str(end - start), 's')
        
        cv2.imshow('测试结果',image_np)
        
        k = cv2.waitKey(0) & 0xFF
        if k == 27:         # 按下esc时，退出
            cv2.destroyAllWindows()
        elif k == ord('s'): # 按下s键时保存并退出
            cv2.imwrite('tet_output.jpg', image_np)
            cv2.destroyAllWindows()


 


