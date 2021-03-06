#if _MSC_VER >= 1600
#pragma execution_character_set("utf-8")
#endif
#ifndef YOLO5_H
#define YOLO5_H
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include<time.h>
using namespace cv;
using namespace dnn;
using namespace std;

struct Net_config
{
    float confThreshold; // Confidence threshold
    float nmsThreshold;  // Non-maximum suppression threshold
    float objThreshold;  //Object Confidence threshold
    string netname;
};

class YOLO
{
    public:
        YOLO(Net_config config);
        Mat blobimg(Mat& frame);
        void detect(Mat& frame,Mat& blob);
    public:
        const float anchors[3][6] = {{10.0, 13.0, 16.0, 30.0, 33.0, 23.0}, {30.0, 61.0, 62.0, 45.0, 59.0, 119.0},{116.0, 90.0, 156.0, 198.0, 373.0, 326.0}};
        const float stride[3] = { 8.0, 16.0, 32.0 };
//        const string classesFile = "coco.names";
        const string classesFile = "best.names";
        const int inpWidth = 640;
        const int inpHeight = 640;
        float confThreshold;
        float nmsThreshold;
        float objThreshold;

        char netname[20];
        vector<string> classes;
        Net net;
        void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);
        void sigmoid(Mat* out, int length);


};

static inline float sigmoid_x(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}
#endif // YOLO5_H
