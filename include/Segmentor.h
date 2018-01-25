#ifndef SEGMENTOR_H
#define SEGMENTOR_H

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "jwh_util.h"

using namespace cv;

class Segmentor {
   public:
    Segmentor(Mat &channel, cv::Mat polygonMask);
    Mat getMask();

   private:
    Mat mChannel;
    Mat mMask;
    Mat mPolygonMask;
    Mat segmentGrabCut();
};

#endif
