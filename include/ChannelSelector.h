#ifndef CHANNEL_SELECTOR_H
#define CHANNEL_SELECTOR_H

#include <iostream>
#include <list>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "ChannelStat.h"
#include "jwh_util.h"

using namespace std;
using namespace cv;
using namespace JWH;

const static float range[] = {0, 256};
const static float *histRange = {range};
const static int histSize = 256;
const static bool uniform = true;
const static bool accumulate_hist = false;

class ChannelSelector {
   public:
    ChannelSelector();
    void create(shared_ptr<vector<String>> filenames,
                shared_ptr<vector<Mat>> imageMats,
                shared_ptr<vector<Mat>> materialIdMats);
    void selectUniqueChannel(Mat *polygon, vector<pair<float, int>> &ordering,
                             float &total_energy);
    void selectRelativeChangePatches(Mat mask, Mat beforeEdit, Mat afterEdit,
                                     vector<Mat> &labels);

   private:
    // variables
    shared_ptr<vector<String>> mFilenames;
    shared_ptr<vector<Mat>> mImageMats;  // exr images, first image input rgb
    shared_ptr<vector<Mat>> mMaterialIdMats;  // exr images,
    vector<Mat> mIntensityImages;
    vector<Mat> mGradientImages;
    vector<Mat> mCandidatePatches;
    Mat mDepthImage;
    String mFolder;
    // Functions
    void init();
    void getNeighbouringPatchStats(vector<PatchStat> &neighbouringPatches,
                                   const vector<Mat> &patchMasks,
                                   const vector<PatchStat> &patchStats,
                                   Mat *polygon, int channelId);
    void getNeighbouringPatches(JWH::Rectangle &rect,
                                vector<cv::Rect> &sampleRectangles);
    /*    void getDepthHistograms(vector<cv::Rect> &sampleRectangles,*/
    // vector<Mat> &depthRectangleHistograms,
    /*int &histogram_size);*/
    void computeDistanceMatrixes(vector<Mat> &distanceMatrices,
                                 vector<cv::Rect> &sampleRectangles,
                                 vector<Mat> &depthRectangleHistograms,
                                 int &histogram_size);

    void computeDistanceChanges(
        vector<Mat> &neighbouringPatches, Mat beforeEdit, Mat afterEdit,
        std::list<std::pair<double, int>> &distance_change_list);

    void findNeighbouringPatches(vector<Mat> &neighbouringPatches, Mat polygon);
};

// Util functions
double outlierScore(Mat &distanceMatrix);
bool sort_pair(const std::pair<double, int> &lhs,
               const std::pair<double, int> &rhs);
void findOutlier(vector<Mat> distanceMatrices,
                 std::list<std::pair<double, int>> &score_list);

#endif
