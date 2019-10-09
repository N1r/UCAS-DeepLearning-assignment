# -*- coding: utf-8 -*-

import numpy as np
import os
import tensorflow as tf
import cv2
import sys
import argparse

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util



#添加python环境变量
CWD_PATH = os.getcwd()
sys.path.append(CWD_PATH)
SLIM_PATH = os.path.join(CWD_PATH,'slim')
sys.path.append(SLIM_PATH)

#已经训练好的模型路径
MODEL_NAME = 'pedestrian_detection'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join(CWD_PATH,'label_map.pbtxt')

NUM_CLASSES = 1  #分类数目

#载入label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def detect_objects(image_np, sess, detection_graph):
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')


    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # 开始检测
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # 可视化识别结果
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    
    return image_np

#检测路径是否正确
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError("请检查路径是否正确，路径中'\\'用'/'代替")

#cwd = os.getcwd()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream', dest="stream", type=int, 
                        default=0, help='视频流方式：0为采集摄像头视频流，1为采集本地视频文件。')
    parser.add_argument('--video_dir', dest='video_dir', type=dir_path,
                        default=None, help='视频文件路径，如果stream为1，则此项必须指定。')
    parser.add_argument('--output_dir', dest='output_dir', type=dir_path,
                        default=None, help='输出视频路径，如果不指定，则不保存输出视频。')
    args = parser.parse_args()
    
    #判断视频流方式
    if args.stream == 0:
        cap=cv2.VideoCapture(0)
    elif args.stream == 1:
        if args.video_dir == None:
            raise Exception('未指定文件路径！')
        else :
            cap=cv2.VideoCapture(args.video_dir)
    else :
        raise Exception('请重新指定视频流方式：0为采集摄像头视频流，1为采集本地视频文件。')
        
    #判断是否输出检测结果
    is_output = False
    if args.output_dir != None:
        is_output = True
        filename = args.output_dir  #输出路径
        codec = cv2.VideoWriter_fourcc('m','p','4','v') #输出视频格式
        framerate = int(cap.get(cv2.CAP_PROP_FPS)) #输出帧率
        WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #输出视频宽度
        HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #输出视频高度
        resolution = (WIDTH,HEIGHT)
        VideoFileOutput = cv2.VideoWriter(filename,codec,framerate, resolution)
    #将模型载入内存中
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            
            
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:   
            ret=True
            while (ret):
                ret, image_np=cap.read()
                image_np = detect_objects(image_np,sess,detection_graph)
                
                if is_output == True:
                    VideoFileOutput.write(image_np)
                cv2.imshow('实时检测',image_np)
                if cv2.waitKey(25) & 0xFF==ord('q'):
                    break
    cv2.destroyAllWindows()
    cap.release()
