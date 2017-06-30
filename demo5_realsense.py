import sys

sys.path.append('./')

from yolo.net.yolo_tiny_net import YoloTinyNet 
import tensorflow as tf 
import cv2
import numpy as np
import rospy
import sys
import struct
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import yaml
from matplotlib import pyplot as plt
from geometry_msgs.msg import Twist
classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

twist = Twist()



class map_img_to_world:
    def __init__(self):

        self.br = CvBridge()

        # If you subscribe /camera/depth_registered/hw_registered/image_rect topic, the depth image and rgb image are 
        # already registered. So you don't need to call register_depth_to_rgb()
        self.depth_image_sub = rospy.Subscriber("/camera/depth_registered/hw_registered/image_rect",Image,self.depth_callback)
        self.rgb_image_sub = rospy.Subscriber("/camera/rgb/image_rect_color",Image,self.rgb_callback)
        self.cmd_vel_sub=rospy.Subscriber("/cmd_vel",Twist,self.cmd_callback)	
        self.ir_img = None
        self.rgb_img = None

        self.rgb_rmat = None
        self.rgb_tvec = None
        self.ir_rmat = None
        self.ir_tvec = None

        self.ir_to_rgb_rmat = None
        self.ir_to_rgb_tvec = None
        self.depth_image = None
        self.rgb_image = None
        self.ir_to_world_tvec = None
        self.ir_to_rgb_rmat = None
        self.depth_image = None
        self.rgb_image = None
        self.rgb_img=None
        self.count = 0
        self.drawing = False # true if mouse is pressed
        self.rect_done = False
        self.ix1 = -1
        self.iy1 = -1
        self.ix2 = -1
        self.iy2 = -1
        self.depth_img=None

        cv2.namedWindow('RGB Image')

    
    def depth_callback(self,data):
        try:
            self.depth_image= self.br.imgmsg_to_cv2(data, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)
        # print "depth"

        depth_min = np.nanmin(self.depth_image)
        depth_max = np.nanmax(self.depth_image)


        self.depth_img = self.depth_image.copy()
        self.depth_img[np.isnan(self.depth_image)] = depth_min
        self.depth_img = ((self.depth_img ) / (5) * 255).astype(np.uint8)
        #print (data)
        #cv2.imshow("Depth Image", depth_img)
        #cv2.waitKey(5)
        # stream = open("/home/chentao/depth_test.yaml", "w")
        # data = {'img':depth_img.tolist()}
        # yaml.dump(data, stream)


    def rgb_callback(self,data):
        try:
            self.rgb_image = self.br.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        #cv2.imshow('RGB Image', self.rgb_image)
        #cv2.waitKey(5)
        self.rect_done = True
        #return self.rgb_image
        # print "rgb"
    def cmd_callback(self,data):
        global twist
        twist.linear.x=data.linear.x
        twist.angular.z=data.angular.z
        #print (sys.getsizeof(data.linear.x))
        file=open("/home/robot/data/cmd.bat","a")
        bytes=struct.pack('f',twist.linear.x)
        file.write(bytes)
        file.close()
        #cv2.imshow('RGB Image', self.rgb_image)
        #cv2.waitKey(5)
        #self.rect_done = True
        #return self.rgb_image
        # print "rgb"

    def get_rgb (self):
       
        return  self.rgb_image

    def get_depth (self):
       
        return  self.depth_img







def process_predicts(predicts):
  p_classes = predicts[0, :, :, 0:20]
  C = predicts[0, :, :, 20:22]
  coordinate = predicts[0, :, :, 22:]

  p_classes = np.reshape(p_classes, (7, 7, 1, 20))
  C = np.reshape(C, (7, 7, 2, 1))

  P = C * p_classes

  #print P[5,1, 0, :]

  index = np.argmax(P)

  index = np.unravel_index(index, P.shape)

  class_num = index[3]

  coordinate = np.reshape(coordinate, (7, 7, 2, 4))

  max_coordinate = coordinate[index[0], index[1], index[2], :]

  xcenter = max_coordinate[0]
  ycenter = max_coordinate[1]
  w = max_coordinate[2]
  h = max_coordinate[3]

  xcenter = (index[1] + xcenter) * (448/7.0)
  ycenter = (index[0] + ycenter) * (448/7.0)

  w = w * 448/2
  h = h * 448/2

  xmin = xcenter - w/3.0
  ymin = ycenter - h/3.0

  xmax = xmin + w
  ymax = ymin + h

  return xmin, ymin, xmax, ymax, class_num
if __name__ == "__main__":
    rospy.init_node('map_img_to_world')
    ic = map_img_to_world()
    common_params = {'image_size': 448, 'num_classes': 20, 'batch_size':1}
    net_params = {'cell_size': 7, 'boxes_per_cell':2, 'weight_decay': 0.0005}

    net = YoloTinyNet(common_params, net_params, test=True)

    image = tf.placeholder(tf.float32, (1, 448, 448, 3))
    predicts = net.inference(image)
    try:
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            if ic.rect_done:
                point_rgb = ic.get_rgb()
                point_depth=ic.get_depth()
                


                sess = tf.Session()

                np_img = point_rgb
                resized_img = cv2.resize(np_img, (448, 448))
                resized_img_depth=cv2.resize(point_depth,(448,448))

                np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)


                np_img = np_img.astype(np.float32)

                np_img = np_img / 255.0 * 2 - 1
                np_img = np.reshape(np_img, (1, 448, 448, 3))

                saver = tf.train.Saver(net.trainable_collection)

                saver.restore(sess, 'models/pretrain/yolo_tiny.ckpt')

                np_predict = sess.run(predicts, feed_dict={image: np_img})

                xmin, ymin, xmax, ymax, class_num = process_predicts(np_predict)
                class_name = classes_name[class_num]
                cv2.rectangle(resized_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))
                cv2.putText(resized_img, class_name, (int(xmin), int(ymin)), 2, 1.5, (0, 0, 255))
                cv2.rectangle(resized_img_depth, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))
                cv2.putText(resized_img_depth, class_name, (int(xmin), int(ymin)), 2, 1.5, (0, 0, 255))
                #cv2.imwrite('cat_out.jpg', resized_img)
                cv2.imshow('RGB Image', resized_img)
                cv2.imshow('depth Image', resized_img_depth)
                cv2.waitKey(1)
                image_point_x=(xmin+xmax)/2;
                image_point_y=(ymin+ymax)/2;
                depth_1=resized_img_depth[int(image_point_x)][int(image_point_y)]
                depth_2=resized_img_depth[int(image_point_x)+5][int(image_point_y)+5]
                depth_3=resized_img_depth[int(image_point_x)+5][int(image_point_y)-5]
                depth_4=resized_img_depth[int(image_point_x)-5][int(image_point_y)+5]
                depth_5=resized_img_depth[int(image_point_x)-5][int(image_point_y)-5]
                depth_center=(depth_1+depth_2+depth_3+depth_4+depth_5)/5
                print(depth_center)
                #if(twist.linear.x!=0)or(twist.angular.z!=0):
                    #if(classes_name[class_num]=="person"):
                        #print("ok")
                        #file=open("/home/robot/data/linear.bat","a")
                        #bytes=struct.pack('f',twist.linear.x)
                        #file.write(bytes)
                        #file.close() 

                        #file=open("/home/robot/data/angular.bat","a")
                        #bytes=struct.pack('f', twist.angular.z)
                        #file.write(bytes)
                        #file.close()

                        #file=open("/home/robot/data/rgb.bat","wb")
                        #np.save(file,resized_img)
                        #file.close()

                        #file=open("/home/robot/data/depth.bat","wb")
                        #np.save(file,resized_img_depth)
                        #file.close()

                
               
                sess.close()
            rate.sleep()

    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

