#include "ros/ros.h"

#include <openpose/flags.hpp>
#include <openpose/headers.hpp>

class MyOPWrapper{

};

int main(int argc, char **argv){
    ros::init(argc, argv, "openpose_node");
    ros::NodeHandle n;

    while(ros::ok())
    {
        ROS_INFO_STREAM("OK");
    }
    return 0;
}