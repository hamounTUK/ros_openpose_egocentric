#!/bin/bash


PATH_TO_OP="${1:- "/home/$USER/git/openpose/"}"

ln -s -f $PATH_TO_OP ~/.ros/op

source ../../devel/setup.bash

roslaunch ros_openpose_egocentric detect_body_rosbag.launch 

