#ifndef CHANNEL_STAT_H
#define CHANNEL_STAT_H

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace Eigen;
using namespace cv;

class MaskEdit;

struct PatchStat {
    PatchStat()
        : mean(0, 0, 0)
        , stddev(0, 0, 0)
        , gradient_mean(0, 0, 0)
        , depth_mean(0, 0, 0)
        , centroid(0, 0)
    {
    }
    Vector3f mean;
    Vector3f stddev;
    Vector3f gradient_mean;
    Vector3f log_mean;
    Vector3f depth_mean;
    Vector2i centroid;
};

class ChannelStat {
   public:
    ChannelStat(vector<Vector3f> means, vector<Vector3f> stddevs,
                String filename);
    static ChannelStat computeChannelStatistics(String filename,
                                                Mat channel_input,
                                                Mat material_id, Mat edit_mask);
    vector<Vector3f> means() { return mMeans; }
    vector<Vector3f> stddevs() { return mStddevs; }
    String filename() { return mFilename; }

   private:
    String mFilename;
    vector<Vector3f> mMeans;
    vector<Vector3f> mStddevs;
};

namespace EditEmbedding {
extern Mat computeEmbedding(pair<Eigen::Vector3f, Eigen::Vector3f> edit_means,
                            vector<ChannelStat> &channelStats);

extern PatchStat computeNeightbourStatistics(Mat channel, Mat mask,
                                             Mat material);
}

#endif
