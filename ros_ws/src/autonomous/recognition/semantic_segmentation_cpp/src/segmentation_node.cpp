// -*- lsst-c++ -*-
#include <semantic_segmentation_cpp/segmentation_node.hpp>

/**
 * @brief Timer class constructor.
 * 
 * @param[in] name String represents the name of the code to be timed
 */
Timer::Timer(std::string name):name(name){
    this->t_start = std::chrono::system_clock::now();
    this->t_stop = std::chrono::system_clock::now();
    this->t_average = 0;
    this->t_itertion = 0;
    this->t_total_time = 0;
    this->idle= true;
}

/**
 * @brief Timer class destructor.
 *
 * @details Timer class to time code execution
 * 
 * @param[in] name String represents the name of the code to be timed
 */
Timer::~Timer(){
    auto num = this->getAverage();
    std::cout<<this->name<<": "<<num/1000<<" ms"<<std::endl;
}


/**
 * @class Timer
 * 
 * @brief A function to start the timer.
 * 
 */
void Timer::start(){
    if (this->idle){
    this->t_start = std::chrono::system_clock::now();
    this->idle = false;
    }
}


/**
 * @class Timer
 * 
 * @brief A function to stop the timer.
 * 
 */
void Timer::stop(){
    if(!this->idle){
    this->t_stop = std::chrono::system_clock::now();
    this->t_total_time +=  duration_cast<microseconds>(this->t_stop - this->t_start).count(); 
    ++ this->t_itertion;
    this->idle = true;
    }
}


/**
 * @class Timer
 * 
 * @brief A function to calculate the average time.
 * 
 */
double Timer::getAverage(){
    this->t_average = this->t_total_time/this->t_itertion;
    return this->t_average; 
}


// pipeline timer declaration
Timer t_pipeline("Average pipeline time"), t_tocv("Average from rosmsg to cvimage");


/**
 * @brief Segmenation node class constructor.
 * 
 * @param[in] img_h (uint) represents the height of the image.
 * 
 * @param[in] img_w (uint) represents the width of the image.
 * 
 * @param[in] classes (uint) represents the number of classes. 
 */
SegNode::SegNode(): img_h(376), img_w(672),classes(3)
{
    // Set the node handle pointer
    this->it.reset(new image_transport::ImageTransport(this->nh));
    this->loaded = false;
    
    // Set computation rate 
    float compute_rate = 30;

    // Set publishing rate
    float publish_rate = 30;
    
    // Prepare timers inputs
    ros::Duration compute_duration = ros::Duration((1/compute_rate));
    ros::Duration publish_duration = ros::Duration((1/publish_rate));
    ros::Duration load_duration = ros::Duration((1/5));
    
    // Publishers
    this->pub =  this->nh.advertise<std_msgs::UInt8MultiArray>(SEG_OUTPUT_TOPIC, 1);
    this->rgb_pub = this->it->advertise(SEG_RGB_OUTOUT_TOPIC, 1);
    
    // Subscriber
    this->sub = this->it->subscribe(CAMERA_TOPIC, 1,&SegNode::cameraCallback, this);
    
    // Timers
    this->compute_t = this->nh.createTimer(compute_duration, &SegNode::compute,this);
    this->publish_t = this->nh.createTimer(publish_duration, &SegNode::publishMessage,this);
    this->load_t = this->nh.createTimer(load_duration, &SegNode::loadModel,this, true); // one-shot timer to load the model
     
    // Set tensor options
    this->options = torch::TensorOptions().dtype(at::kByte).device(torch::kCPU).requires_grad(false);

    // Set all pointers to nullptr.
    this->seg_output_ptr = nullptr; 
    this->input_cv_ptr = nullptr;
    this->output_rgb_ptr = nullptr;
    
    //Initialize colors
    colors.resize(this->classes, std::vector<uint8_t>(3));
    colors[0] = std::vector<uint8_t>{0,0,255};
    colors[1] = std::vector<uint8_t>{0,255,0};
    colors[2] = std::vector<uint8_t>{255,0,255};
    


    // Prepare output layout
    std::vector<std_msgs::MultiArrayDimension> dims;
    dims.resize(2);
    dims[0].label = "height";
    dims[0].size = img_h;
    dims[0].stride = img_h * img_w;
    dims[1].label = "width";
    dims[1].size = img_w;
    dims[1].stride = img_w;
    this->layout.dim = dims;
}

/**
 * Callback function associated with the camera subscriber to get a new message
 *
 * Callback function invoked when a new message is published on the camera output topic
 * to get a new message.
 * 
 * @param[in] image_msg Pointer to the image message
 */
void SegNode::cameraCallback(const sensor_msgs::ImageConstPtr& image_msg){
    if(this->loaded){
    t_tocv.start();
    // copy the received image data(use opencv copy function not share)
    this->input_cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::RGB8); 
    t_tocv.stop();
    }
}

/**
 * Timer callback function to load the jit model
 *
 * Function to load the jit model. Instead of loading the model while initilizing the node, 
 * the function is associated with a one-shot timer to load the model. 
 * 
 * @param[in] event Structure passed as a parameter to the callback invoked by a ros::Timer
 */
void SegNode::loadModel(const ros::TimerEvent& event){
    // load torch script
    this->model = torch::jit::load(MODEL_PATH, torch::kCUDA);
    this->loaded = true;

}


/**
 * Class member function.
 * 
 * @class SegNode
 * 
 * @brief Computes the output and prepare output variables to be published
 * 
 * @param[in] event Structure passed as a parameter to the callback invoked by a ros::Timer
 */
void SegNode::compute(const ros::TimerEvent& event){
    
    try{
        if(this->input_cv_ptr !=nullptr && this->loaded){
        
        t_pipeline.start();
        
        at::Tensor tensor_image = torch::from_blob((this->input_cv_ptr->image.data), 
        { this->input_cv_ptr->image.rows, this->input_cv_ptr->image.cols, this->input_cv_ptr->image.channels() }, 
        this->options).permute({2,0,1}).toType(at::kFloat).div_(255.).unsqueeze_(0).to(at::kCUDA);
        
        
        std::vector<torch::jit::IValue> jit_inputs;
        jit_inputs.push_back(tensor_image); // copy
        
        at::Tensor output = this->model.forward(jit_inputs).toTensor().argmax(1).to(at::kCPU).to(torch::kUInt8);
        
        this->fillOutput(output);
        torch::cuda::synchronize(); 
        t_pipeline.stop();
        }
        
    }
    catch (cv_bridge::Exception& e){
         ROS_ERROR("cv_bridge exception: %s", e.what());
         return;
       }

}


/**
 * Class member function.
 * 
 * @class SegNode
 * 
 * @brief Helper function to fill the outputs
 * 
 * @param[in] event Structure passed as a parameter to the callback invoked by a ros::Timer
 */
void SegNode::fillOutput(at::Tensor& input){
    
    if (this->seg_output_ptr == nullptr);
    {
        this->seg_output_ptr =  &(this->seg_output);
        this->seg_output_ptr->layout = this->layout;
        //this->seg_output_ptr->data.reserve(input.numel());
    }

    // fill seg output (ids)
    this->seg_output_ptr->data.clear(); 
    this->seg_output_ptr->data.assign(input.data_ptr<uchar>(), input.data_ptr<uchar>()+input.numel());
    
    // fill seg output (rgb)
    auto frame = this->toCvImage(input);
    auto rgb_frame = this->toRGBMap(frame);
    this->output_rgb_ptr =  cv_bridge::CvImage(std_msgs::Header(), "rgb8", rgb_frame).toImageMsg();

}


/**
 * Class member function
 * 
 * @class SegNode
 * 
 * @brief Function that publishs output messages.
 * 
 * @param[in] event Structure passed as a parameter to the callback invoked by a ros::Timer
 */
void SegNode::publishMessage(const ros::TimerEvent& event){
    // publish rgb image and output images
    if(this->loaded){
    //t_publishing.start();
    if(this->output_rgb_ptr !=nullptr)
    {
        this->rgb_pub.publish(this->output_rgb_ptr);
    }
    //if(this->seg_output_ptr !=nullptr)
    if (this->seg_output_ptr !=nullptr)
    {
        this->pub.publish(*(this->seg_output_ptr));
    }
    //std::cout<<"HIHIHIIH"<<std::endl;
    //t_publishing.stop();
    }
}


/**
 * Class member function
 * 
 * @class SegNode
 * 
 * @brief Function that converts a tensor to cv-image.
 * 
 * @details Function that converts a singe channel tensor image to a single channel cv-image.
 * 
 * @param[in] event Structure passed as a parameter to the callback invoked by a ros::Timer
 */
cv::Mat SegNode::toCvImage(at::Tensor& tensor){
        tensor = tensor.squeeze();
        tensor = tensor.contiguous();
        //tensor = tensor.mul(255).clamp(0, 255).to(torch::kU8);
        int16_t height = tensor.size(0);
        int16_t width = tensor.size(1);
        cv::Mat mat = cv::Mat(cv::Size(width, height), CV_8UC1, tensor.data_ptr<uchar>());
        return mat.clone(); 
}


/**
 * Class member function
 * 
 * @class SegNode
 * 
 * @brief Function that converts a 1-channel cv-image to 3-channels cv-image.
 * 
 * @details Function that converts a singe channel cv image to a three channels cv-image using a color map. 
 * 
 * @param[in] event Structure passed as a parameter to the callback invoked by a ros::Timer
 */
cv::Mat SegNode::toRGBMap(cv::Mat& cv_image) {
    // cv_image has one channel
    cv::Mat r,g,b, output;
    r = cv::Mat::zeros(cv::Size(cv_image.cols, cv_image.rows), CV_8UC1);
    g = cv::Mat::zeros(cv::Size(cv_image.cols, cv_image.rows), CV_8UC1);
    b = cv::Mat::zeros(cv::Size(cv_image.cols, cv_image.rows), CV_8UC1);
    for (size_t i = 0; i < this->classes; ++i) {
        auto mask = cv_image==i;
        r.setTo(this->colors[i][0], mask);     // red
        g.setTo(this->colors[i][1], mask);    // green
        b.setTo(this->colors[i][2], mask);     // bllue
    }
    cv::merge(std::vector<cv::Mat>{r,g,b}, output);
    return output;
    
}


// main function
int main(int argc, char **argv)
{
    ros::init(argc, argv, "segmentation node", ros::init_options::AnonymousName);
    SegNode nd_ex;
    ros::spin();
    return EXIT_SUCCESS;

}


