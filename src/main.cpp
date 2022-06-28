#include "ros/ros.h"

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <openpose/flags.hpp>
#include <openpose/headers.hpp>
#include <opencv2/opencv.hpp>
#define OPENPOSE_FLAGS_DISABLE_POSE


#include <actionlib/server/simple_action_server.h>

#include <user_defined_msgs/BodyDetectedActionAction.h>
#include <user_defined_msgs/BodyDetectedActionActionFeedback.h>
#include <user_defined_msgs/BodyDetectedActionActionGoal.h>
#include <user_defined_msgs/BodyDetectedActionActionResult.h>
#include <user_defined_msgs/BodyDetectedActionFeedback.h>
#include <user_defined_msgs/BodyDetectedActionGoal.h>
#include <user_defined_msgs/BodyDetectedActionResult.h>


typedef actionlib::SimpleActionServer<user_defined_msgs::BodyDetectedActionAction> Server;

const int LEFT_ELBOW = 6;
const int LEFT_WRIST = 7;
const int RIGHT_ELBOW = 3;
const int RIGHT_WRIST = 4;



void configureWrapper(op::Wrapper& opWrapper)
{
    try
    {
        // Configuring OpenPose

        // logging_level
        op::checkBool(
            0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
            __LINE__, __FUNCTION__, __FILE__);
        op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
        op::Profiler::setDefaultX(FLAGS_profile_speed);

        // Applying user defined configuration - GFlags to program variables
        // outputSize
        const auto outputSize = op::flagsToPoint(op::String(FLAGS_output_resolution), "-1x-1");
        // netInputSize
        const auto netInputSize = op::flagsToPoint(op::String(FLAGS_net_resolution), "-1x368");
        // faceNetInputSize
        const auto faceNetInputSize = op::flagsToPoint(op::String(FLAGS_face_net_resolution), "368x368 (multiples of 16)");
        // handNetInputSize
        const auto handNetInputSize = op::flagsToPoint(op::String(FLAGS_hand_net_resolution), "368x368 (multiples of 16)");
        // poseMode
        const auto poseMode = op::flagsToPoseMode(FLAGS_body);
        // poseModel
        const auto poseModel = op::flagsToPoseModel(op::String(FLAGS_model_pose));
        // JSON saving
        if (!FLAGS_write_keypoint.empty())
            op::opLog(
                "Flag `write_keypoint` is deprecated and will eventually be removed. Please, use `write_json`"
                " instead.", op::Priority::Max);
        // keypointScaleMode
        const auto keypointScaleMode = op::flagsToScaleMode(FLAGS_keypoint_scale);
        // heatmaps to add
        const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg,
                                                      FLAGS_heatmaps_add_PAFs);
        const auto heatMapScaleMode = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
        // >1 camera view?
        const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1);
        // Face and hand detectors
        const auto faceDetector = op::flagsToDetector(FLAGS_face_detector);
        const auto handDetector = op::flagsToDetector(FLAGS_hand_detector);
        // Enabling Google Logging
        const bool enableGoogleLogging = true;

        // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
        const op::WrapperStructPose wrapperStructPose{
            poseMode, netInputSize, FLAGS_net_resolution_dynamic, outputSize, keypointScaleMode, FLAGS_num_gpu,
            FLAGS_num_gpu_start, FLAGS_scale_number, (float)FLAGS_scale_gap,
            op::flagsToRenderMode(FLAGS_render_pose, multipleView), poseModel, !FLAGS_disable_blending,
            (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap, FLAGS_part_to_show, op::String(FLAGS_model_folder),
            heatMapTypes, heatMapScaleMode, FLAGS_part_candidates, (float)FLAGS_render_threshold,
            FLAGS_number_people_max, FLAGS_maximize_positives, FLAGS_fps_max, op::String(FLAGS_prototxt_path),
            op::String(FLAGS_caffemodel_path), (float)FLAGS_upsampling_ratio, enableGoogleLogging};
        opWrapper.configure(wrapperStructPose);
        // Face configuration (use op::WrapperStructFace{} to disable it)
        const op::WrapperStructFace wrapperStructFace{
            FLAGS_face, faceDetector, faceNetInputSize,
            op::flagsToRenderMode(FLAGS_face_render, multipleView, FLAGS_render_pose),
            (float)FLAGS_face_alpha_pose, (float)FLAGS_face_alpha_heatmap, (float)FLAGS_face_render_threshold};
        opWrapper.configure(wrapperStructFace);
        // Hand configuration (use op::WrapperStructHand{} to disable it)
        const op::WrapperStructHand wrapperStructHand{
            FLAGS_hand, handDetector, handNetInputSize, FLAGS_hand_scale_number, (float)FLAGS_hand_scale_range,
            op::flagsToRenderMode(FLAGS_hand_render, multipleView, FLAGS_render_pose), (float)FLAGS_hand_alpha_pose,
            (float)FLAGS_hand_alpha_heatmap, (float)FLAGS_hand_render_threshold};
        opWrapper.configure(wrapperStructHand);
        // Extra functionality configuration (use op::WrapperStructExtra{} to disable it)
        const op::WrapperStructExtra wrapperStructExtra{
            FLAGS_3d, FLAGS_3d_min_views, FLAGS_identification, FLAGS_tracking, FLAGS_ik_threads};
        opWrapper.configure(wrapperStructExtra);
        // Output (comment or use default argument to disable any output)
        const op::WrapperStructOutput wrapperStructOutput{
            FLAGS_cli_verbose, op::String(FLAGS_write_keypoint), op::stringToDataFormat(FLAGS_write_keypoint_format),
            op::String(FLAGS_write_json), op::String(FLAGS_write_coco_json), FLAGS_write_coco_json_variants,
            FLAGS_write_coco_json_variant, op::String(FLAGS_write_images), op::String(FLAGS_write_images_format),
            op::String(FLAGS_write_video), FLAGS_write_video_fps, FLAGS_write_video_with_audio,
            op::String(FLAGS_write_heatmaps), op::String(FLAGS_write_heatmaps_format), op::String(FLAGS_write_video_3d),
            op::String(FLAGS_write_video_adam), op::String(FLAGS_write_bvh), op::String(FLAGS_udp_host),
            op::String(FLAGS_udp_port)};
        opWrapper.configure(wrapperStructOutput);
        // No GUI. Equivalent to: opWrapper.configure(op::WrapperStructGui{});
        // Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
        if (FLAGS_disable_multi_thread)
            opWrapper.disableMultiThreading();
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}


void execute_cb(const user_defined_msgs::BodyDetectedActionGoalConstPtr& goal, Server* as)
{
    ROS_WARN_STREAM("Goal recieved");

}


class ActionServerManager{
    protected:
        ros::NodeHandle nh;
        Server server_;
        op::Wrapper& opw_;
    public:
        ActionServerManager(op::Wrapper& opw):
            opw_(opw),
            server_(nh, "/body", boost::bind(&ActionServerManager::execute_cb, this, _1), false)
        {
            opw_.start();
            server_.start();
            ROS_INFO_STREAM("Action Server started...");
        }

        void execute_cb(const user_defined_msgs::BodyDetectedActionGoalConstPtr &goal){

            // sensor_msgs::ImageConstPtr fisheye1 = ros::topic::waitForMessage<sensor_msgs::Image>("/t265/fisheye1/image_raw", ros::Duration(0.0)) ;


            sensor_msgs::Image fisheye1 = goal->fisheye1;
            ROS_ERROR_STREAM("Goal recieved. Image captured!  " << fisheye1.header.seq);

        
            cv_bridge::CvImagePtr cv_ptr;
            try
            {
              cv_ptr = cv_bridge::toCvCopy(fisheye1, sensor_msgs::image_encodings::BGR8);
            }
            catch (cv_bridge::Exception& e)
            {
              ROS_ERROR("cv_bridge exception: %s", e.what());
              return;
            }

            // const cv::Mat cvImageToProcess = cv::imread("frame0460.jpg");
            const op::Matrix imageToProcess = OP_CV2OPCONSTMAT(cv_ptr->image);
            auto datumProcessed = opw_.emplaceAndPop(imageToProcess);
            
            auto keypoints = datumProcessed->at(0)->poseKeypoints;
            // ROS_INFO_STREAM(keypoints.getSize(1));
            int num_peeple = keypoints.getSize(0);
            int num_body_part = keypoints.getSize(1);
            int rwx = 0, rwy = 0;


            int max = 0;
            int index = 0;
                            ROS_WARN_STREAM("OKKKK2222");

            for(int i = 0; i < num_peeple; i++){

                const int left_wrist_baseIndex = keypoints.getSize(2)*(i*num_body_part + LEFT_WRIST);
                ROS_WARN_STREAM("OKKKK");
                if (max < keypoints[left_wrist_baseIndex + 2]){
                    max = keypoints[left_wrist_baseIndex + 2];
                    index = i;
                }
            }


            for(int i = 0; i < num_peeple; i++){
                i = index;    
                const auto left_elbow_baseIndex = keypoints.getSize(2)*(index*num_body_part + LEFT_ELBOW);
                const auto left_wrist_baseIndex = keypoints.getSize(2)*(index*num_body_part + LEFT_WRIST);
                const auto right_elbow_baseIndex = keypoints.getSize(2)*(index*num_body_part + RIGHT_ELBOW);
                const auto right_wrist_baseIndex = keypoints.getSize(2)*(index*num_body_part + RIGHT_WRIST);
                rwx = keypoints[left_wrist_baseIndex];
                rwy = keypoints[left_wrist_baseIndex + 1];
            }

            user_defined_msgs::BodyDetectedActionResult result;
            user_defined_msgs::EgocentricBodySegmentDetected msg;
            
            msg.left_elbow_x = (int)fisheye1.header.seq;

            result.detected_segments.push_back(msg); 

            server_.setSucceeded(result);

            cv::circle(cv_ptr->image, cv::Point(rwx, rwy), 2, cv::Scalar(0, 255, 0), 2);
            cv::imshow("win", cv_ptr->image);
            cv::waitKey(3);
            
            ROS_ERROR_STREAM("Goal Processed.################" << fisheye1.header.seq);


        }

    
};



int main(int argc, char **argv){
    ros::init(argc, argv, "openpose_node");
    ros::NodeHandle nh;
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    ros::AsyncSpinner spinner(0);
    spinner.start();

    op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};
    configureWrapper(opWrapper);
    // opWrapper.start();
    
    ActionServerManager action_server(opWrapper);

    // Server server(nh, "/body", boost::bind(&execute_cb, _1, &server), false);
    // server.start();


    
    // opWrapper.exec();

    // const cv::Mat cvImageToProcess = cv::imread("frame0460.jpg");
    // const op::Matrix imageToProcess = OP_CV2OPCONSTMAT(cvImageToProcess);
    // auto datumProcessed = opWrapper.emplaceAndPop(imageToProcess);
 
    // if (datumProcessed != nullptr)
    // {
    //     ROS_WARN_STREAM("Found $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
    //     const auto& poseKeypoints = datumProcessed->at(0)->poseKeypoints;


    //     const auto numberPeopleDetected = poseKeypoints.getSize(0);
    //     ROS_ERROR_STREAM(numberPeopleDetected);
    //     ROS_INFO_STREAM(poseKeypoints);



    //     if (datumProcessed != nullptr && !datumProcessed->empty())
    //     {
    //         ROS_WARN_STREAM("Passed this, goint to show");
    //         // Display image
    //         const cv::Mat cvMat = OP_OP2CVCONSTMAT(datumProcessed->at(0)->cvOutputData);
    //         if (!cvMat.empty())
    //         {   
    //             cv::imshow(OPEN_POSE_NAME_AND_VERSION + " - Tutorial C++ API", cvMat);
    //             cv::waitKey(0);
    //         }
    //     }
    // }
    // else
    //     ROS_WARN_STREAM("No process or no result");
    ros::waitForShutdown();
    return 0;
}