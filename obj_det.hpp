//#include <opencv2/core/utils/filesystem.hpp>


#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <librealsense2/rs.hpp>

#ifdef CV_CXX11
#include <mutex>
#include <thread>
#include <queue>
#include <string> 
#endif

using namespace cv;
using namespace dnn;
using namespace rs2;

float confThreshold, nmsThreshold;
std::vector<std::string> classes;

int width = 1280;
int height = 720;
int fps = 15;

void preprocess(const Mat& frame, Net& net, Size inpSize, float scale,
                       const Scalar& mean, bool swapRB);

Mat postprocess(Mat& frame, const std::vector<Mat>& out, Net& net, rs2::points& points);
Mat postprocessInfraRight(Mat& frame, const std::vector<Mat>& outs, Net& net, rs2::points& points);
Mat postprocessInfraLeft(Mat& frame, const std::vector<Mat>& outs, Net& net, rs2::points& points);

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

void callback(int pos, void* userdata);

cv::Mat frame_to_mat(const rs2::frame& f);

std::string genArgument(const std::string& argName, const std::string& help,
                        const std::string& modelName, const std::string& zooFile,
                        char key = ' ', std::string defaultVal = "");

std::string genPreprocArguments(const std::string& modelName, const std::string& zooFile);

Mat createResultImg(Mat img, Mat gray, Rect box, rs2::points& points);
Net loadNet();

pipeline startCamera();

int detect(bool oneImage, Mat& result_l, Mat& result_r, Net net, pipeline pipe);

//std::string findFile(const std::string& filename);

#ifdef CV_CXX11
template <typename T>
class QueueFPS : public std::queue<T>
{
public:
    QueueFPS() : counter(0) {}

    void push(const T& entry)
    {
        std::lock_guard<std::mutex> lock(mutex);

        std::queue<T>::push(entry);
        counter += 1;
        if (counter == 1)
        {
            // Start counting from a second frame (warmup).
            tm.reset();
            tm.start();
        }
    }

    T get()
    {
        std::lock_guard<std::mutex> lock(mutex);
        T entry = this->front();
        this->pop();
        return entry;
    }

    float getFPS()
    {
        tm.stop();
        double fps = counter / tm.getTimeSec();
        tm.start();
        return static_cast<float>(fps);
    }

    void clear()
    {
        std::lock_guard<std::mutex> lock(mutex);
        while (!this->empty())
            this->pop();
    }

    unsigned int counter;

private:
    TickMeter tm;
    std::mutex mutex;
};
#endif  // CV_CXX11

Net loadNet(){
    const std::string weigthsFile = "/home/labarea/Scrivania/CODICI/testYolo4HoleGrayCpp/data/hole_2000.weights";
    const std::string cfgFile = "/home/labarea/Scrivania/CODICI/testYolo4HoleGrayCpp/data/hole.cfg";
    const std::string classesFile = "/home/labarea/Scrivania/CODICI/testYolo4HoleGrayCpp/data/hole.names";

    std::string modelPath = weigthsFile;
    std::string configPath = cfgFile;

    int backend = 0;
    int target = 0;

    // Open file with classes names.
    std::ifstream ifs(classesFile.c_str());
    if (!ifs.is_open())
        CV_Error(Error::StsError, "File " + classesFile + " not found");
    std::string line;
    while (std::getline(ifs, line)) {
        classes.push_back(line);
    }
       
    // Load a model.
    Net net = readNet(modelPath, configPath, "");
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    //Stampa nomi dei layers
    /*std::vector<std::string> names = net.getLayerNames();
    for(int i =0; i < names.size(); i++){
        std::cout << names[i] << std::endl;
    }*/

    return net;
}

pipeline startCamera()
{
    pipeline pipe;
    rs2::config config;

    config.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_RGB8, fps);//.as<video_stream_profile>();
    config.enable_stream(RS2_STREAM_INFRARED, 1, width, height, RS2_FORMAT_Y8, fps);
    config.enable_stream(RS2_STREAM_INFRARED, 2, width, height, RS2_FORMAT_Y8, fps);
    config.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, fps);
    //rs2::align align_to(RS2_STREAM_COLOR);
    
    rs2::pipeline_profile pipeline_profile = pipe.start(config);

    rs2::device dev = pipeline_profile.get_device();

    auto depth_sensor = dev.first<rs2::depth_sensor>();
    if (depth_sensor.supports(RS2_OPTION_EXPOSURE)) {
        depth_sensor.set_option(RS2_OPTION_EXPOSURE, 30000.f); //10715
    }
    if (depth_sensor.supports(RS2_OPTION_GAIN)) {
        depth_sensor.set_option(RS2_OPTION_GAIN, 148.f); //33
    }
    if (depth_sensor.supports(RS2_OPTION_LASER_POWER)){
        depth_sensor.set_option(RS2_OPTION_LASER_POWER, 30); //30
    }
    if (depth_sensor.supports(RS2_OPTION_EMITTER_ENABLED)) {
        depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 0.f);
    }

    return pipe;
}

int detect(bool oneImage, Mat& result_l, Mat& result_r, Net net, pipeline pipe)
{

    confThreshold = 0.80; //parser.get<float>("thr");
    nmsThreshold = 0.40; //parser.get<float>("nms");
    float scale = 0.00392; //parser.get<float>("scale");
    Scalar mean = Scalar(0,0,0,0);//parser.get<Scalar>("mean");
    bool swapRB = false;//parser.get<bool>("rgb");
    int inpWidth = 416;//parser.get<int>("width");
    int inpHeight = 416;//parser.get<int>("height");
    size_t asyncNumReq = 0; //parser.get<int>("async");

    std::string filename_lnn;
    std::string filename_rnn;
    std::string filename_l;
    std::string filename_r;
    
    std::vector<String> outNames = net.getUnconnectedOutLayersNames();

    
    // Capture 30 frames to give autoexposure, etc. a chance to settle
    for (int k = 0; k < 30; ++k) pipe.wait_for_frames();


    bool process = true;
    int i = 1;
    char c;

    //VideoWriter video_left("left.avi", VideoWriter::fourcc('M','J','P','G'),1, Size(width, height));
    //VideoWriter video_right("right.avi", VideoWriter::fourcc('M','J','P','G'),1, Size(width, height));
    //VideoWriter video_color("color.avi", VideoWriter::fourcc('M','J','P','G'),1, Size(width, height));

    rs2::pointcloud pc;
    rs2::points points;
    //rs2::align align_to_depth(RS2_STREAM_DEPTH);
    //rs2::align align_to_color(RS2_STREAM_COLOR);

    while(process){
        auto data = pipe.wait_for_frames();
        //data = align_to_color.process(data);
        //data = align_to_color.process(data);
        auto color_frame = data.get_color_frame();
        auto depth_frame = data.get_depth_frame();
        rs2::video_frame ir_frame_left = data.get_infrared_frame(1);
        rs2::video_frame ir_frame_right = data.get_infrared_frame(2);
        
        //pc.map_to(color_frame);
        //points = pc.calculate(depth_frame);
        
        //std::cout << "Line 222";
        //COLOR
    
        auto frame = frame_to_mat(color_frame);
        cv::Mat gray, gray_temp;
        cv::cvtColor(frame, gray, COLOR_RGB2GRAY); //Convert in grayScale
        cv::cvtColor(gray, gray_temp, COLOR_GRAY2RGB); //Convert in "grayScale with 3channel"
        //auto depth = frame_to_mat(depth_frame);
        //preprocess(frame, net, Size(inpWidth, inpHeight), scale, mean, swapRB);
        preprocess(gray_temp, net, Size(inpWidth, inpHeight), scale, mean, swapRB);
        std::vector<Mat> outs;
        net.forward(outs, outNames);
        postprocess(gray_temp, outs, net, points);
        //video_color.write(frame);
        imshow("Image", gray_temp);
        waitKey(15);
     
    /*
        //INFRA LEFT
        cv::Mat dMat_left = cv::Mat(cv::Size(width, height), CV_8UC1, (void*)ir_frame_left.get_data());
        cv::Mat frame_il;
        cv::cvtColor(dMat_left, frame_il, COLOR_GRAY2RGB);
        preprocess(frame_il, net, Size(inpWidth, inpHeight), scale, mean, swapRB);
        std::vector<Mat> outs;
        net.forward(outs, outNames);
        result_l = postprocess(frame_il, outs, net, points);
     
       
        //INFRA Right
        cv::Mat dMat_right = cv::Mat(cv::Size(width, height), CV_8UC1, (void*)ir_frame_right.get_data());
        cv::Mat frame_ir;
        cv::cvtColor(dMat_right, frame_ir, COLOR_GRAY2RGB);
        preprocess(frame_ir, net, Size(inpWidth, inpHeight), scale, mean, swapRB);
        //std::vector<Mat> outs;
        net.forward(outs, outNames);
        result_r = postprocess(frame_ir, outs, net, points); //postprocessInfraRight(frame_ir, outs, net, points);

        
        //video_left.write(frame_il);
        imshow("Left", frame_il);
        //video_right.write(frame_ir);
        imshow("Right", frame_ir);
*/
    /*
        c = waitKey(1);
        if (c == 's') {
          filename_lnn = "result/holeL_" + std::to_string(i) + ".tif";
          filename_rnn = "result/holeR_" + std::to_string(i) + ".tif";
          //filename_l = "result/imL_" + std::to_string(i) + ".tif";
          //filename_r = "result/imR_" + std::to_string(i) + ".tif";
          cv::imwrite(filename_lnn, result_l);
          cv::imwrite(filename_rnn, result_r);
          //cv::imwrite(filename_l, dMat_left);
          //cv::imwrite(filename_r, dMat_right);
          std::cout << "Saved " << filename_lnn << std::endl;
          std::cout << "Saved " << filename_rnn << std::endl;
          std::cout << "Saved " << filename_l << std::endl;
          std::cout << "Saved " << filename_r << std::endl; 
          i++;
        }
        else if (c == 'q')
          break;
*/
        
        if(oneImage){
            process = false;
            waitKey();
        }
    }
    //video_left.release();
    //video_right.release();
    //video_color.release();

    std::cout << "Exit" << std::endl;
    return 0;
}

void preprocess(const Mat& frame, Net& net, Size inpSize, float scale,
                       const Scalar& mean, bool swapRB)
{
    static Mat blob;
    // Create a 4D blob from a frame.
    if (inpSize.width <= 0) inpSize.width = frame.cols;
    if (inpSize.height <= 0) inpSize.height = frame.rows;
    blobFromImage(frame, blob, 1.0, inpSize, Scalar(), swapRB, false, CV_8U);

    // Run a model.
    net.setInput(blob, "", scale, mean);
    if (net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
    {
        resize(frame, frame, inpSize);
        Mat imInfo = (Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
        net.setInput(imInfo, "im_info");
    }
}

Mat postprocess(Mat& frame, const std::vector<Mat>& outs, Net& net, rs2::points& points)
{
    static std::vector<int> outLayers = net.getUnconnectedOutLayers();
    static std::string outLayerType = net.getLayer(outLayers[0])->type;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect> boxes;
    if (outLayerType == "Region")
    {
        for (size_t i = 0; i < outs.size(); ++i)
        {
            // Network produces output blob with a shape NxC where N is a number of
            // detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold)
                {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }
    }
    else{
        CV_Error(Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);
    }


    std::vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    Mat img_gray;
    //img_gray = imread("/home/labarea-franka/Documents/yolo_test/test2/build/white.png");
    //img_gray = imread("/home/labarea-franka/Documents/yolo_test/test2/build/gray.png");
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        //img_gray = createResultImg(frame, img_gray, box, points);
        drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame);
    }
    return frame;
}

Mat postprocessInfraRight(Mat& frame, const std::vector<Mat>& outs, Net& net, rs2::points& points)
{
    static std::vector<int> outLayers = net.getUnconnectedOutLayers();
    static std::string outLayerType = net.getLayer(outLayers[0])->type;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect> boxes;
    if (outLayerType == "Region")
    {
        for (size_t i = 0; i < outs.size(); ++i)
        {
            // Network produces output blob with a shape NxC where N is a number of
            // detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold)
                {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left-25, top, width, height));
                }
            }
        }
    }
    else{
        CV_Error(Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);
    }

    std::vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    Mat img_gray;
    img_gray = imread("/home/labarea-franka/Documents/yolo_test/test2/build/white.png");
    //img_gray = imread("/home/labarea-franka/Documents/yolo_test/test2/build/gray.png");
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        cv::waitKey(0);
        img_gray = createResultImg(frame, img_gray, box, points);
        drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame);
    }
    return img_gray;
}

Mat postprocessInfraLeft(Mat& frame, const std::vector<Mat>& outs, Net& net, rs2::points& points)
{
    static std::vector<int> outLayers = net.getUnconnectedOutLayers();
    static std::string outLayerType = net.getLayer(outLayers[0])->type;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect> boxes;
    if (outLayerType == "Region")
    {
        for (size_t i = 0; i < outs.size(); ++i)
        {
            // Network produces output blob with a shape NxC where N is a number of
            // detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold)
                {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }
    }
    else{
        CV_Error(Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);
    }

    std::vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    Mat img_gray;
    img_gray = imread("/home/labarea-franka/Documents/yolo_test/test2/build/white.png");
    //img_gray = imread("/home/labarea-franka/Documents/yolo_test/test2/build/gray.png");
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        img_gray = createResultImg(frame, img_gray, box, points);
        drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame);
    }
    return img_gray;
}

Mat createResultImg(Mat img, Mat gray, Rect box, rs2::points& points)
{
    //auto pVertices = points.get_vertices();
    //auto pVertices = (void *)pVerticesTmp;

    int w = img.cols;
    int h = img.rows;
    int left = box.x;
    int top = box.y;
    int right = box.x + box.width;
    int bottom = box.y + box.height;


    int n = (720+240);
    int l = left/(n*1.0)*h;
    int r = right/(n*1.0)*h;
    int min = (w < h ) ? w : h;
    //std::cout << " SIZE: " << pVertices[350].x << " " << pVertices[250].y << " ---" << std::endl;

    //std::cout << img.size << std::endl;
/*    for (int k = 0; k < points.size(); ++k) {
        //if(top  < pVertices[k].y && pVertices[k].y < bottom){
          //  if(l < pVertices[k].x && pVertices[k].x < r ){
                gray.at<int>(pVertices[k].x,pVertices[k].y) = img.at<int>(pVertices[k].x,pVertices[k].y);
            //} 
        //}
    }
*/
    for(int i = 0; i < 720; i++){
        for(int j = 0; j < (720+249); j++){
            if(top  < i && i < bottom){
                if(l < j && j < r ){
                    gray.at<int>(i,j) = img.at<int>(i,j);
                } 
            }     
        }
    }

    return gray;
}

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));

    std::string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ": " + label;
    }

    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height),
              Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}

void callback(int pos, void*)
{
    confThreshold = pos * 0.01f;
}

cv::Mat frame_to_mat(const rs2::frame& f)
{
    using namespace cv;
    using namespace rs2;

    auto vf = f.as<video_frame>();
    const int w = vf.get_width();
    const int h = vf.get_height();

    if (f.get_profile().format() == RS2_FORMAT_BGR8)
    {
        return Mat(Size(w, h), CV_8UC3, (void*)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_RGB8)
    {
        auto r = Mat(Size(w, h), CV_8UC3, (void*)f.get_data(), Mat::AUTO_STEP);
        cvtColor(r, r, COLOR_RGB2BGR);
        return r;
    }
    else if (f.get_profile().format() == RS2_FORMAT_Z16)
    {
        return Mat(Size(w, h), CV_16UC1, (void*)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_Y8)
    {
        return Mat(Size(w, h), CV_8UC1, (void*)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_DISPARITY32)
    {
        return Mat(Size(w, h), CV_32FC1, (void*)f.get_data(), Mat::AUTO_STEP);
    }

    throw std::runtime_error("Frame format is not supported yet!");
}


std::string genArgument(const std::string& argName, const std::string& help,
                        const std::string& modelName, const std::string& zooFile,
                        char key, std::string defaultVal)
{
    if (!modelName.empty())
    {
        FileStorage fs(zooFile, FileStorage::READ);
        if (fs.isOpened())
        {
            FileNode node = fs[modelName];
            if (!node.empty())
            {
                FileNode value = node[argName];
                if (!value.empty())
                {
                    if (value.isReal())
                        defaultVal = format("%f", (float)value);
                    else if (value.isString())
                        defaultVal = (std::string)value;
                    else if (value.isInt())
                        defaultVal = format("%d", (int)value);
                    else if (value.isSeq())
                    {
                        for (size_t i = 0; i < value.size(); ++i)
                        {
                            FileNode v = value[(int)i];
                            if (v.isInt())
                                defaultVal += format("%d ", (int)v);
                            else if (v.isReal())
                                defaultVal += format("%f ", (float)v);
                            else
                              CV_Error(Error::StsNotImplemented, "Unexpected value format");
                        }
                    }
                    else
                        CV_Error(Error::StsNotImplemented, "Unexpected field format");
                }
            }
        }
    }
    return "{ " + argName + " " + key + " | " + defaultVal + " | " + help + " }";
}

/*
std::string findFile(const std::string& filename)
{
    if (filename.empty() || utils::fs::exists(filename))
        return filename;

    const char* extraPaths[] = {getenv("OPENCV_DNN_TEST_DATA_PATH"),
                                getenv("OPENCV_TEST_DATA_PATH")};
    for (int i = 0; i < 2; ++i)
    {
        if (extraPaths[i] == NULL)
            continue;
        std::string absPath = utils::fs::join(extraPaths[i], utils::fs::join("dnn", filename));
        if (utils::fs::exists(absPath))
            return absPath;
    }
    CV_Error(Error::StsObjectNotFound, "File " + filename + " not found! "
             "Please specify a path to /opencv_extra/testdata in OPENCV_DNN_TEST_DATA_PATH "
             "environment variable or pass a full path to model.");
}*/

std::string genPreprocArguments(const std::string& modelName, const std::string& zooFile)
{
    return genArgument("model", "Path to a binary file of model contains trained weights. "
                                "It could be a file with extensions .caffemodel (Caffe), "
                                ".pb (TensorFlow), .t7 or .net (Torch), .weights (Darknet), .bin (OpenVINO).",
                       modelName, zooFile, 'm') +
           genArgument("config", "Path to a text file of model contains network configuration. "
                                 "It could be a file with extensions .prototxt (Caffe), .pbtxt (TensorFlow), .cfg (Darknet), .xml (OpenVINO).",
                       modelName, zooFile, 'c') +
           genArgument("mean", "Preprocess input image by subtracting mean values. Mean values should be in BGR order and delimited by spaces.",
                       modelName, zooFile) +
           genArgument("scale", "Preprocess input image by multiplying on a scale factor.",
                       modelName, zooFile, ' ', "1.0") +
           genArgument("width", "Preprocess input image by resizing to a specific width.",
                       modelName, zooFile, ' ', "-1") +
           genArgument("height", "Preprocess input image by resizing to a specific height.",
                       modelName, zooFile, ' ', "-1") +
           genArgument("rgb", "Indicate that model works with RGB input images instead BGR ones.",
                       modelName, zooFile);
}
