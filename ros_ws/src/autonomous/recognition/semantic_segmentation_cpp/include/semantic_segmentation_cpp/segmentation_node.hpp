#ifndef EXMP_NODE_H
#define EXMP_NODE_H

// Types
#ifndef HAVE_IPL
   typedef unsigned char uchar;
   typedef unsigned short ushort;
#endif

// ROS headers
#include <ros/ros.h>
#include <ros/time.h>
#include <std_msgs/UInt8MultiArray.h>
#include <std_msgs/MultiArrayLayout.h>
#include <std_msgs/MultiArrayDimension.h>

// ROS-OpenCV Headers
#include <image_transport/image_transport.h> //image
#include <cv_bridge/cv_bridge.h>

// Torch headers
#include <torch/torch.h>
#include <torch/script.h> // One-stop header.
#include <torch/cuda.h>
using namespace torch::indexing;

// C++ Std Headers 
#include <iostream>
#include <memory>
#include <vector>
#include <cstdint>
#include <opencv2/imgproc/imgproc.hpp>


// Timer Header
#include <chrono>
using namespace std::chrono;


// Topics names
std::string CAMERA_TOPIC = "/racer/camera1/image_raw";
std::string SEG_RGB_OUTOUT_TOPIC = "/cpp_seg_output/rgb";
std::string SEG_OUTPUT_TOPIC = "/cpp_seg_output/num";

// Model path
std::string MODEL_PATH = "/home/hatem/ros-torch-exp/ros_ws/models/test_model_traced_gpu.pt";


// Segmentation node
class SegNode
{
    private:
    // variable to indicate that the model is loaded
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

    // Helper function to fill the output variables
    void fillOutput(at::Tensor&);
    
    // Function to convert a tensor image to CV image
    cv::Mat toCvImage(at::Tensor&);

    // Function to convert a 1-channel segmentation output to 3-channels colored image
    cv::Mat toRGBMap(cv::Mat&);
    std::vector<std::vector<uint8_t>> colors;

    public:
    // Constructor
    SegNode();

    // Image dimensions
    uint16_t img_h, img_w;

    // number of classes
    uint8_t classes;
    
    // Destructor
    //~SegNode();

    // Publisher callback function associated with a timer
    void publishMessage(const ros::TimerEvent&);
    //void cameraCallback(const sensor_msgs::ImageConstPtr& image_msg,
      //                        const sensor_msgs::CameraInfoConstPtr& info_msg);
    
    // Subscriber callback function
    void cameraCallback(const sensor_msgs::ImageConstPtr& image_msg);

    // Callback function associated with a timer
    void compute(const ros::TimerEvent&);

    // Callback function associated with an one-shot timer
    void loadModel(const ros::TimerEvent&);
     
};

// Timer class to time code execution
class Timer{
  private:
  std::chrono::time_point<std::chrono::system_clock> t_start;
  std::chrono::time_point<std::chrono::system_clock> t_stop;
  double t_average, t_itertion, t_total_time;
  std::string name;
  bool idle;
  public:
  Timer(std::string);
  ~Timer();
  double getAverage();
  void start();
  void stop();

};

#endif // EXMP_NODE_H
