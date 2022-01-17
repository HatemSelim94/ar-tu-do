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
/*
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
}*/

Timer t_to_tensor("Average to_tensor conversion time"), t_processing_time("Average processing time"), t_output_time("Average output preparation time"), t_publishing("Average publishing time"), t_color_map("Average Color Map time");

SegNode::SegNode(): img_h(376), img_w(672),classes(3)
{
    this->it.reset(new image_transport::ImageTransport(this->nh));
    this->loaded = false;
    float compute_rate = 30;
    float publish_rate = 30;
    ros::Duration compute_duration = ros::Duration((1/compute_rate));
    ros::Duration publish_duration = ros::Duration((1/publish_rate));
    ros::Duration load_duration = ros::Duration((1/5));
    this->pub =  this->nh.advertise<std_msgs::UInt8MultiArray>(SEG_OUTPUT_TOPIC, 1);
    this->rgb_pub = this->it->advertise(SEG_RGB_OUTOUT_TOPIC, 1);
    this->sub = this->it->subscribe(CAMERA_TOPIC, 1,&SegNode::cameraCallback, this);
    this->compute_t = this->nh.createTimer(compute_duration, &SegNode::compute,this);
    this->publish_t = this->nh.createTimer(publish_duration, &SegNode::publishMessage,this);
    this->load_t = this->nh.createTimer(load_duration, &SegNode::loadModel,this, true); // one-shot timer to load the model
     
    //this->model.to(at::kCUDA);
    this->options = torch::TensorOptions().dtype(at::kByte).device(torch::kCPU).requires_grad(false);

    // Create a vector of inputs.
    this->seg_output_ptr = nullptr; 
    this->input_cv_ptr = nullptr;
    this->output_rgb_ptr = nullptr;
    
    //colors
    colors.resize(this->classes, std::vector<uint8_t>(3));
    colors[0] = std::vector<uint8_t>{0,0,255};
    colors[1] = std::vector<uint8_t>{0,255,0};
    colors[2] = std::vector<uint8_t>{255,0,255};
    


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
    if(this->loaded){
    this->input_cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::RGB8); //(copy not a share)
    //auto conversion_stop = high_resolution_clock::now();
    //auto duration = duration_cast<milliseconds>(conversion_stop - start);
    //std::cout<<"Conversion to cv image rgb time: "<<duration.count()/1000<<"ms"<<std::endl;
    }
}

void SegNode::loadModel(const ros::TimerEvent& event){
    // load torch script
    this->model = torch::jit::load(MODEL_PATH, torch::kCUDA);
    this->loaded = true;

}


// compute the output
void SegNode::compute(const ros::TimerEvent& event){
    //auto start = high_resolution_clock::now();
    
    try{
        if(this->input_cv_ptr !=nullptr && this->loaded)
        {
        t_to_tensor.start();
        t_processing_time.start();
        at::Tensor tensor_image = torch::from_blob((this->input_cv_ptr->image.data), { this->input_cv_ptr->image.rows, this->input_cv_ptr->image.cols, this->input_cv_ptr->image.channels() }, this->options).permute({2,0,1}).toType(at::kFloat).div_(255.).unsqueeze_(0).to(at::kCUDA);
        //std::cout<<"tensor shape: "<<tensor_image.sizes()<<std::endl;
        std::vector<torch::jit::IValue> jit_inputs;
        jit_inputs.push_back(tensor_image); // copy
        //auto conversion_stop = high_resolution_clock::now();
        //auto duration = duration_cast<microseconds>(conversion_stop - start);
        t_to_tensor.stop();
        at::Tensor output = this->model.forward(jit_inputs).toTensor().argmax(1).to(at::kCPU).to(torch::kUInt8);
        //auto out = torch::_unique(output);
        //std::cout<<"Unique Values: "<<std::get<0>(out)<<"HHHHH"<<std::get<1>(out)<<std::endl;
        //at::Tensor output = this->model.forward(jit_inputs).toTensor();
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
    // this->colorMap(); fill this->output_rgb_ptr
    // 
    //###this->output_rgb_ptr = this->input_cv_ptr;
    auto frame = this->toCvImage(input);
    t_color_map.start();
    auto rgb_frame = this->toRGBMap(frame);
    t_color_map.stop();
    this->output_rgb_ptr =  cv_bridge::CvImage(std_msgs::Header(), "rgb8", rgb_frame).toImageMsg();

}

// publish the outputs
void SegNode::publishMessage(const ros::TimerEvent& event){
    // publish rgb image and output images
    if(this->loaded){
    t_publishing.start();
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
    t_publishing.stop();
    }
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

cv::Mat SegNode::toCvImage(at::Tensor& tensor){
        tensor = tensor.squeeze();
        tensor = tensor.contiguous();
        //tensor = tensor.mul(255).clamp(0, 255).to(torch::kU8);
        int16_t height = tensor.size(0);
        int16_t width = tensor.size(1);
        cv::Mat mat = cv::Mat(cv::Size(width, height), CV_8UC1, tensor.data_ptr<uchar>());
        return mat.clone(); 
}

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
    //std::cout<<"HIHIHIHIHIHIHIHIHIH"<<std::endl;
    //std::cout<<"cols"<<output.cols<<"rows"<<output.rows<<" "<<output.channels()<<std::endl;
    return output;
    
}



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


