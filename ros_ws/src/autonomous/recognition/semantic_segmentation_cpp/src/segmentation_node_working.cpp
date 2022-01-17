#include <semantic_segmentation_cpp/segmentation_node.hpp>

Timer::Timer(std::string name):name(name){
    this->t_start = std::chrono::system_clock::now();
    this->t_stop = std::chrono::system_clock::now();
    this->t_average = 0;
    this->t_itertion = 0;
    this->t_total_time = 0;
}
Timer::~Timer(){
    auto num = this->getAverage();
    std::cout<<this->name<<": "<<num/1000<<" ms"<<std::endl;
}

void Timer::start(){
    this->t_start = std::chrono::system_clock::now();
}

void Timer::stop(){
    this->t_stop = std::chrono::system_clock::now();
    this->t_total_time +=  duration_cast<microseconds>(this->t_stop - this->t_start).count(); 
    ++ this->t_itertion;
}

double Timer::getAverage(){
    this->t_average = this->t_total_time/this->t_itertion;
    return this->t_average; 
}
cv::Mat TensorCv::toCvImage(){
    auto size = this->sizes();
    uint16_t height = *(size.end()-2);
    uint16_t width = *(size.end()-1); 
    try
    {
        cv::Mat output_mat(cv::Size{ width, height}, CV_8UC3, this->data_ptr<uint8_t>());
        return output_mat.clone();
    }
    catch (const c10::Error& e)
    {
        std::cout << "an error has occured : " << e.msg() << std::endl;
    }
    return cv::Mat(height, width, CV_8UC3);
}

Timer t_to_tensor("Average to_tensor conversion time"), t_processing_time("Average processing time"), t_output_time("Average output preparation time"), t_publishing("Average publishing time");

SegNode::SegNode(): img_h(376), img_w(672)
{
    this->it.reset(new image_transport::ImageTransport(this->nh));
    float compute_rate = 60;
    float publish_rate = 60;
    ros::Duration compute_duration = ros::Duration((1/compute_rate));
    ros::Duration publish_duration = ros::Duration((1/publish_rate));
    this->pub =  this->nh.advertise<std_msgs::UInt8MultiArray>(SEG_OUTPUT_TOPIC, 1);
    this->rgb_pub = this->it->advertise(SEG_RGB_OUTOUT_TOPIC, 1);
    this->sub = this->it->subscribe(CAMERA_TOPIC, 1,&SegNode::cameraCallback, this);
    this->compute_t = this->nh.createTimer(compute_duration, &SegNode::compute,this);
    this->publish_t = this->nh.createTimer(publish_duration, &SegNode::publishMessage,this);
    // load torch script
    this->model = torch::jit::load("/home/hatem/ros-torch-exp/ros_ws/models/test_model_traced.pt");
    this->options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU).requires_grad(false);

    // Create a vector of inputs.
    this->seg_output_ptr = nullptr; 
    this->input_cv_ptr = nullptr;
    this->output_cv_ptr = nullptr;
    

    // output layout
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

// store the image in a member var
void SegNode::cameraCallback(const sensor_msgs::ImageConstPtr& image_msg){
    //auto start = high_resolution_clock::now();
    this->input_cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::RGB8); //(copy not a share)
    //auto conversion_stop = high_resolution_clock::now();
    //auto duration = duration_cast<milliseconds>(conversion_stop - start);
    //std::cout<<"Conversion to cv image rgb time: "<<duration.count()/1000<<"ms"<<std::endl;
}


// compute the output
void SegNode::compute(const ros::TimerEvent& event)
{
    //auto start = high_resolution_clock::now();
    
    try{
        if(this->input_cv_ptr !=nullptr)
        {
        t_to_tensor.start();
        t_processing_time.start();
        at::Tensor tensor_image = torch::from_blob((this->input_cv_ptr->image.data), { this->input_cv_ptr->image.rows, this->input_cv_ptr->image.cols, this->input_cv_ptr->image.channels() }, this->options).permute({2,0,1}).to(torch::kFloat32);
        //std::cout<<"tensor shape: "<<tensor_image.sizes()<<std::endl;
        tensor_image.unsqueeze_(0);
        std::vector<torch::jit::IValue> jit_inputs;
        jit_inputs.push_back(tensor_image); // copy
        //auto conversion_stop = high_resolution_clock::now();
        //auto duration = duration_cast<microseconds>(conversion_stop - start);
        t_to_tensor.stop();
        at::Tensor output = this->model.forward(jit_inputs).toTensor().argmax(1).to(torch::kUInt8);
        t_processing_time.stop();
        //at::Tensor output = this->model.forward(jit_inputs).toTensor();
        t_output_time.start();
        this->fillOutput(output);
        t_output_time.stop(); 
        //
        }
        

    }
    catch (cv_bridge::Exception& e){
         ROS_ERROR("cv_bridge exception: %s", e.what());
         return;
       }
    //auto stop = high_resolution_clock::now();
    //auto duration = duration_cast<microseconds>(stop - start);
    //std::cout<<"Processing time: "<<duration.count()/1000<<"ms"<<std::endl;
}

// helper function to fill the output vars
void SegNode::fillOutput(at::Tensor& input){
    if (this->seg_output_ptr == nullptr);
    {
        this->seg_output_ptr =  &(this->seg_output);
        this->seg_output_ptr->layout = this->layout;
        //this->seg_output_ptr->data.reserve(input.numel());
    }
    // fill seg output (ids)
    //std::vector<uchar> v(input.data_ptr<uchar>(), input.data_ptr<uchar>() + input.numel());
    this->seg_output_ptr->data.clear(); //
    this->seg_output_ptr->data.assign(input.data_ptr<uchar>(), input.data_ptr<uchar>()+input.numel());

    // fill seg output (rgb)
    //this->input
    // this->colorMap(); fill this->output_cv_ptr
    // 
    this->output_cv_ptr = this->input_cv_ptr;

}

// publish the outputs
void SegNode::publishMessage(const ros::TimerEvent& event){
    // publish rgb image and output images
    t_publishing.start();
    if(this->output_cv_ptr !=nullptr)
    {
        this->rgb_pub.publish(this->output_cv_ptr->toImageMsg());
    }
    //if(this->seg_output_ptr !=nullptr)
    if (this->seg_output_ptr !=nullptr)
    {
        this->pub.publish(*(this->seg_output_ptr));
    }
    //std::cout<<"HIHIHIIH"<<std::endl;
    t_publishing.stop();
}

void SegNode::colorMap(at::Tensor& input, at::Tensor& output){
    
}


// direct conversion
/*
void SegNode::cameraCallback(const sensor_msgs::ImageConstPtr& image_msg)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).requires_grad(false);
    auto img_tensor = torch::from_blob(const_cast<uchar*>(&(image_msg->data[0])),{image_msg->height, image_msg->width, 3},{(image_msg->height)*(image_msg->width)*3,(image_msg->width)*3, 3},options).permute({2,0,1}).unsqueeze(0);
    std::cout<<"tensor shape: "<<img_tensor.sizes()<<std::endl;
    std::cout<<"width: "<<image_msg->width<<" height: "<<image_msg->height<<" step: "<<image_msg->step<<std::endl;
    std::cout<<"image encoding: "<<image_msg->encoding<<" bigendian: "<<image_msg->is_bigendian<<std::endl;
    /*
    auto img_tens_sl = img_tensor.index({"...", "...", Slice(None, 224), Slice(None, 224)});
    std::vector<torch::jit::IValue> jit_inputs;
    jit_inputs.push_back(img_tens_sl);
    at::Tensor output = this->model.forward(jit_inputs).toTensor();
    */
    //std::cout<<"okok\n";
//}

//cv::Mat SegNode::toCvImage(at::Tensor tensor)
//{
    
//}



int main(int argc, char **argv)
{
    ros::init(argc, argv, "exmp_node", ros::init_options::AnonymousName);
    //try{
    SegNode nd_ex;
    //}
    /*
    catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
    }
    */
   //torch::
    ros::spin();
    return EXIT_SUCCESS;

}


