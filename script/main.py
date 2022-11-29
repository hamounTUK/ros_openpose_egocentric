import rospy
from sensor_msgs.msg import Image, CameraInfo
import cv2 as cv
import numpy as np
import sys
import tf
from cv_bridge import CvBridge
import math

try:
    sys.path.append('op/build/python')
    from openpose import pyopenpose as op
except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e    



class EgocentricOP:
    def __init__(self):

        # OpenPose Configuration
        params = dict()
        params["model_folder"] = "op/models/"
        params["heatmaps_add_parts"] = True
        params["net_resolution"] = "320x320"
        # params["heatmaps_scale"] = "1"
        

        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()
        self.datum = op.Datum()
        
        # keypoint index for left and right hand in Body25 body kinematic model(Body25 is the OpenPose default model)
        self.keypoints = [4, 7]

        # transform listener for getting kinematic transformation(not used yet)
        self.listener = tf.TransformListener()

        # ros topic subscribers
        self.fisheye1_cam_info_sub = rospy.Subscriber("/t265/fisheye1/camera_info", CameraInfo, self.fisheye1_cam_info_cb)
        self.fisheye1_sub = rospy.Subscriber("/t265/fisheye1/image_raw_rect", Image, self.fisheye1_cb)

        # from ROS to OpenCV
        self.cv_bridge = CvBridge()


        self.last_chosen_left  = None # a tuple in the form of (row, col)
        self.last_chosen_right = None # a tuple in the form of (row, col)

        self.first_data = True



    # (heatmap, k: number of expecting clusters in each heatmap, ws: window_size, hm_h: heatmap hight, hm_w: heatmap width)    
    def find_top_k(self, top_k, hm, k, ws, hm_h, hm_w):
        # top_k = []
        for i in range(k):
            max = np.amax(hm)
            point = np.where(hm == max)
            r = point[0][0]
            c = point[1][0]
            
            # normalized row and column
            top_k.append((r/hm_h, c/hm_w, max/255))
            
            # set neighborhood of (r,c) to zero within the window size
            try:
                hm[r-ws:r+ws, c-ws:c+ws] = np.zeros((2*ws, 2*ws))
            except:
                pass

        return top_k



    # Get camera parameters
    def fisheye1_cam_info_cb(self, msg):
        self.fisheye1_projection_matrix = np.array(msg.P).reshape((3,4))
        self.fisheye1_cam_info_sub.unregister()


    # Get fisheye1 camera images
    def fisheye1_cb(self, msg):

        # convert images from ROS to OpenCV structure
        img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        img = cv.merge((img,img,img))
        
        #image size
        img_height = msg.height
        img_width = msg.width
        
        # image timestamp
        secs = msg.header.stamp.secs
        nsecs = msg.header.stamp.nsecs

        # initialize the last state
        if self.first_data == True:
            self.last_chosen_left = (img_height -1, 0)
            self.last_chosen_right = (img_height - 1, img_width - 1)
            self.first_data = False

        """
            We can later place codes here to match kinematic values with images.
            So far, only the visual measurements is considered.
        """    

        # feeding image to the OpenPose
        self.datum.cvInputData = img
        self.opWrapper.emplaceAndPop(op.VectorDatum([self.datum]))
        # outputImageF = (self.datum.inputNetData[0].copy())[0,:,:,:] + 0.5
        # outputImageF = cv.merge([outputImageF[0,:,:], outputImageF[1,:,:], outputImageF[2,:,:]])
        # outputImageF = (outputImageF*255.).astype(dtype='uint8')
        heatmaps = self.datum.poseHeatMaps
        heatmaps = (heatmaps).astype(dtype='uint8')

        dim0 = heatmaps.shape[1]
        dim1 = heatmaps.shape[2]

        # setting parameters for post-processing.
        # an score function for finding the best candidate.
        # score = alpha * confidence_score + beta * distance_inertial + gamma * distance_last  
        alpha = 1; beta  = 0; gamma = 0

        #look for k candidates in each heatmap
        k = 2 
        # window size(in the size of heatmap not the actual image) to not detect any other candidate within this window 
        w = int(dim0 / 10) 
        top_candidates_lst = []
        
        for i, keypoint in enumerate(self.keypoints):
            hm = heatmaps[keypoint, : , : ].copy()
            # list of tuples which each touple is in the form of (row, col, confidence_score), all normalized in [0, 1]
            self.find_top_k(top_candidates_lst, hm, k, w, dim0, dim1) 

        max_score_left = 0; max_score_right = 0

        # distance to inertial(projected), later will be defined
        inertial_dist_l = 1; inertial_dist_r = 1

        # distance to the last selected point for left and right hand
        last_dist_l = 1; last_dist_r = 1

        # selected point (r, c) for this image for left and right hand
        selected_left  = None; selected_right = None 
        
        # cand is a tuple in the form of (row, col, confidence_score)
        for i, cand in enumerate(top_candidates_lst):
            """
                inertial_dist(r and l) should be calculated later
            """
            last_dist_l =  math.sqrt((cand[0] - self.last_chosen_left[0])**2 + (cand[1] - self.last_chosen_left[1])**2)
            last_dist_r =  math.sqrt((cand[0] - self.last_chosen_right[0])**2 + (cand[1] - self.last_chosen_right[1])**2)

            if last_dist_l == 0: 
                last_dist_l = 1

            if last_dist_r == 0:
                last_dist_r = 1

            score_left  = alpha * cand[2] + beta * (1/inertial_dist_l) + gamma * (1/last_dist_l)
            score_right  = alpha * cand[2] + beta * (1/inertial_dist_r) + gamma * (1/last_dist_r)


            if score_left > max_score_left:
                selected_left = (cand[0], cand[1])
                max_score_left = score_left

            
            if score_right > max_score_right:
                selected_right = (cand[0], cand[1])
                max_score_right = score_right


        heatmap = heatmaps[4, :,:].copy()
        # heatmap = heatmaps[7, :,:].copy()

        heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
        # combined = cv.addWeighted(outputImageF, 0.5, heatmap, 0.5, 0)
        cv.imshow("Heatmap", heatmap)

        if selected_left is not None:
            img = cv.circle(img, (int(img_height * selected_left[1]), int(img_width * selected_left[0])), 3, (255,255,255), 3)


        cv.imshow("Fisheye_rect", img)
        cv.waitKey(3)


if __name__ == "__main__":

    rospy.init_node("ros_openpose_egocentric")
    print("ros_openpose_egocentric Node is started!!!")

    op_ego = EgocentricOP()

    rospy.spin()




