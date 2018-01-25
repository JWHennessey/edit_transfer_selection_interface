#ifndef MASK_EDIT_H
#define MASK_EDIT_H
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "ChannelStat.h"
#include "GLTexture.h"
#include "jwh_util.h"

using namespace std;
using namespace cv;

const static int R_GAMMA_INDEX = 0;
const static int R_INPUT_MIN_HIST_INDEX = 1;
const static int R_OUTPUT_MIN_HIST_INDEX = 2;
const static int R_INPUT_MAX_HIST_INDEX = 3;
const static int R_OUTPUT_MAX_HIST_INDEX = 4;

const static int G_GAMMA_INDEX = 5;
const static int G_INPUT_MIN_HIST_INDEX = 6;
const static int G_OUTPUT_MIN_HIST_INDEX = 7;
const static int G_INPUT_MAX_HIST_INDEX = 8;
const static int G_OUTPUT_MAX_HIST_INDEX = 9;

const static int B_GAMMA_INDEX = 10;
const static int B_INPUT_MIN_HIST_INDEX = 11;
const static int B_OUTPUT_MIN_HIST_INDEX = 12;
const static int B_INPUT_MAX_HIST_INDEX = 13;
const static int B_OUTPUT_MAX_HIST_INDEX = 14;

const static int APPLY_MASK_INDEX = 15;
const static int REMOVE_EDIT_INDEX = 16;
const static int RECT_X = 17;
const static int RECT_Y = 18;
const static int RECT_W = 19;
const static int RECT_H = 20;
const static int RATIO_L = 21;
const static int MATERIAL_ID_INDEX = 22;
const static int NO_REL_POS = 23;
const static int REl_POS1_L = 24;
const static int REl_POS1_A = 25;
const static int REl_POS1_B = 26;
const static int REl_POS2_L = 27;
const static int REl_POS2_A = 28;
const static int REl_POS2_B = 29;
const static int REl_POS3_L = 30;
const static int REl_POS3_A = 31;
const static int REl_POS3_B = 32;
const static int REl_POS4_L = 33;
const static int REl_POS4_A = 34;
const static int REl_POS4_B = 35;
const static int CHANNEL_ID1 = 36;
const static int CHANNEL_ID2 = 37;
const static int CHANNEL_ID3 = 38;
const static int CHANNEL_ID4 = 39;
const static int CHANNEL_ID5 = 40;
const static int CHANNEL_ID6 = 41;
const static int CHANNEL_ID7 = 42;
const static int CHANNEL_ID8 = 43;
const static int CHANNEL_ID9 = 44;
const static int CHANNEL_ID10 = 46;
const static int EXPOSURE = 47;

const static int RGB_GAMMA_INDEX = 48;
const static int RGB_INPUT_MIN_HIST_INDEX = 49;
const static int RGB_OUTPUT_MIN_HIST_INDEX = 50;
const static int RGB_INPUT_MAX_HIST_INDEX = 51;
const static int RGB_OUTPUT_MAX_HIST_INDEX = 52;

const static int HUE_INDEX = 53;
const static int SATURATION_INDEX = 54;
const static int LIGHTNESSS_INDEX = 55;

const static int BRIGHTNESS_INDEX = 56;
const static int CONTRAST_INDEX = 57;

const static int BLUR_SIZE_INDEX = 58;
const static int BLUR_SIGMA_INDEX = 59;

class MaskEdit {
   public:
    MaskEdit(Mat mask, JWH::Rectangle rect, String channelName, Mat channel,
             Mat material, int materialIndex);
    MaskEdit(Mat mask, String channelName, Mat params, Mat channel,
             Mat material, int materialIndex, vector<Mat> relativePatches);
    ImageDataType& maskGLData() { return mMaskData; };
    Mat getMask() { return mMask; }
    Mat& getMaskRef() { return mMask; }
    Mat* getMaskPtr() { return &mMask; }
    Mat& getEditParamRef() { return mEditParams; }
    void invertMask();
    float intersectionArea(JWH::Rectangle rect);
    void writeParameters(String filename);
    void transferParameters(vector<Mat> input_channels);
    pair<PatchStat, PatchStat> getEditStats(const Mat& input_channel);


    float setRGBGamma(float gamma);
    float getRGBGamma();
    void setRGBHistMin(float value);
    float getRGBHistMin();
    void setRGBHistMax(float value);
    float getRGBHistMax();
    void setRGBHistOutMin(float value);
    float getRGBHistOutMin();
    void setRGBHistOutMax(float value);
    float getRGBHistOutMax();

    float setRGamma(float gamma);
    float getRGamma();
    void setRHistMin(float value);
    float getRHistMin();
    void setRHistMax(float value);
    float getRHistMax();
    void setRHistOutMin(float value);
    float getRHistOutMin();
    void setRHistOutMax(float value);
    float getRHistOutMax();

    float setGGamma(float gamma);
    float getGGamma();
    void setGHistMin(float value);
    float getGHistMin();
    void setGHistMax(float value);
    float getGHistMax();
    void setGHistOutMin(float value);
    float getGHistOutMin();
    void setGHistOutMax(float value);
    float getGHistOutMax();

    float setBGamma(float gamma);
    float getBGamma();
    void setBHistMin(float value);
    float getBHistMin();
    void setBHistMax(float value);
    float getBHistMax();
    void setBHistOutMin(float value);
    float getBHistOutMin();
    void setBHistOutMax(float value);
    float getBHistOutMax();

    void setHue(float value);
    float getHue();

    void setSaturation(float value);
    float getSaturation();

    void setLightness(float value);
    float getLightness();

    void setBrightness(float value);
    float getBrightness();

    void setContrast(float value);
    float getContrast();
    
    void setExposure(float value);
    float getExposure();

    void setBlurSize(int value);
    int getBlurSize();
    
    void setBlurSigma(float value);
    float getBlurSigma();

    void applyMaskToLayer(bool val);
    bool applyMask();
    // void computeEditRatio(Mat channel);
    void computeRelativePositioning(Mat beforeEdit, Mat afterEdit,
                                    vector<Mat> patches);

    /*     bool setMaskType(float value)*/
    //{
    // return mEditParams.at<float>(0, MASK_TYPE_INDEX) = value;
    /*}*/
    void setRemoveEdit(bool val);
    bool getRemoveEdit();
    JWH::Rectangle getRectangle();
    void toggleMaskParams();

    void setCallback(std::function<void()> func) { mParameterCallback = func; };

    void setChannelIndexes(vector<int> indexes);
    void setChannelIndex(int val) { mChannelIndex = val; }
    void setMaskIndex(int val) { mMaskIndex = val; }
    int getChannelIndex() { return mChannelIndex; }
    int getMaskIndex() { return mMaskIndex; }
    vector<int> getChannelIndexes();  // { return mChannelIndexes; }

    void copySegmentationToMask(Mat segmentation, Mat initialMask);
    void drawMask(Eigen::Vector2f p1, Eigen::Vector2f p2, int size, float intensity);

    void setUpdateTexture(bool val) {mUpdateTexture = val; }
    bool getUpdateTexture() { return mUpdateTexture; }

   private:
    Mat mMask;
    Mat mMaterialId;
    Mat mChannel;
    std::function<void()> mParameterCallback;
    vector<Mat> mRelativePatches;
    vector<Mat> mRelativePatchesDeltas;
    int mMaterialIndex;
    int mChannelIndex;
    int mMaskIndex;
    vector<int> mChannelIndexes;
    Rect mRect;
    String mChannelName;
    Mat mEditParams;       // 40 x 1 vector
    Mat mOldParamsToggle;  // 40 x 1 vector
    ImageDataType mMaskData;
    bool mApplyMask;
    bool mRemoveEdit;
    bool mParamType;
    bool mUpdateTexture;
    // Functions
    void initEditParams();
};

Mat segmentationToFloatImage(Mat segmentation);
PatchStat getPatchStats(Mat channel, Mat patch);

#endif
