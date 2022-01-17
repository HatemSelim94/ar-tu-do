#ifndef EXMP_NODE_H
#define EXMP_NODE_H

// ROS includes
#include <ros/ros.h>
#include <ros/time.h>
#include <std_msgs/UInt8MultiArray.h>
#include <std_msgs/MultiArrayLayout.h>
#include <std_msgs/MultiArrayDimension.h>
#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <torch/script.h> // One-stop header.
#include <vector>
#include <image_transport/image_transport.h> //image
#include <cstdint>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
using namespace torch::indexing;
std::string CAMERA_TOPIC = "/racer/camera1/image_raw";
std::string SEG_RGB_OUTOUT_TOPIC = "/cpp_seg_output/rgb";
std::string SEG_OUTPUT_TOPIC = "/cpp_seg_output/num";
std::string MODEL_PATH = "/home/hatem/ros-torch-exp/ros_ws/models/test_model_traced_gpu.pt";
#ifndef HAVE_IPL
   typedef unsigned char uchar;
   typedef unsigned short ushort;
#endif
#include <chrono>
using namespace std::chrono;
/*
class TensorCv: public at::Tensor{
  cv::Mat toCvImage();
};*/

class SegNode
{
    private:
    bool loaded;
    //pubs
    ros::Publisher pub; // output publisher
    image_transport::Publisher rgb_pub; // output rgb publisher
    //subs
    //image_transport::CameraSubscriber sub;
    image_transport::Subscriber sub;
    //node handle
    ros::NodeHandle nh;
    //image transport
    boost::shared_ptr<image_transport::ImageTransport> it;
    // timers
    ros::Timer publish_t, compute_t, load_t;
    // torch vars
    torch::jit::script::Module model; 
    torch::TensorOptions options;
    // opencv vars
    sensor_msgs::ImagePtr output_rgb_ptr;
    cv_bridge::CvImagePtr input_cv_ptr;
    // seg output
    std_msgs::UInt8MultiArray seg_output;
    std_msgs::UInt8MultiArray * seg_output_ptr;
    std_msgs::MultiArrayLayout layout;

    // fill output member function
    void fillOutput(at::Tensor&);
    void colorMap(at::Tensor&, at::Tensor&);
    
    
    cv::Mat toCvImage(at::Tensor&);
    cv::Mat toRGBMap(cv::Mat&);
    std::vector<std::vector<uint8_t>> colors;

    public:
    // Constructor
    uint16_t img_h, img_w;
    uint8_t classes;
    SegNode();
    // Destructor
    //~SegNode();
    void publishMessage(const ros::TimerEvent&);
    //void cameraCallback(const sensor_msgs::ImageConstPtr& image_msg,
      //                        const sensor_msgs::CameraInfoConstPtr& info_msg);
    void cameraCallback(const sensor_msgs::ImageConstPtr& image_msg);
    void compute(const ros::TimerEvent&);
    void loadModel(const ros::TimerEvent&);
    
    
};


class Timer{
  private:
  std::chrono::time_point<std::chrono::system_clock> t_start;
  std::chrono::time_point<std::chrono::system_clock> t_stop;
  double t_average, t_itertion, t_total_time;
  std::string name;
  public:
  Timer(std::string);
  ~Timer();
  double getAverage();
  void start();
  void stop();

};

#endif // EXMP_NODE_H
