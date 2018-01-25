#include "MaskEdit.h"
#include "ParameterOptimisation.h"

MaskEdit::MaskEdit(Mat mask, JWH::Rectangle rect, String channelName,
                   Mat channel, Mat material, int materialIndex)
    : mMask(segmentationToFloatImage(mask))
    , mRect(JWH::convertToCVRect(rect, mMask))
    , mChannelName(channelName)
    , mEditParams(1, NO_EDIT_PARAMS, CV_32FC1)
    , mApplyMask(true)
    , mRemoveEdit(false)
    , mParamType(false)
    , mChannel(channel)
    , mMaterialId(material)
    , mMaterialIndex(materialIndex)
    , mChannelIndex(-1)
    , mMaskIndex(-1)
    , mUpdateTexture(false)
{
    cout << "MaskEdit start Init " << endl;
    GLTexture texture("Mask_Not_Numbered");
    auto data = texture.load(mMask);
    mMaskData = std::move(make_pair(std::move(texture), std::move(data)));

    initEditParams();
    cout << "MaskEdit end Init " << endl;
}

MaskEdit::MaskEdit(Mat mask, String channelName, Mat params, Mat channel,
                   Mat material, int materialIndex, vector<Mat> relativePatches)
    : mMask(segmentationToFloatImage(mask))
    , mChannelName(channelName)
    , mEditParams(params)
    , mChannel(channel)
    , mMaterialId(material)
    , mMaterialIndex(materialIndex)
    , mRelativePatches(relativePatches)
    , mChannelIndex(-1)
    , mMaskIndex(-1)
    , mUpdateTexture(false)
{
    GLTexture texture("Mask_Not_Numbered");
    auto data = texture.load(mMask);
    mMaskData = std::move(make_pair(std::move(texture), std::move(data)));

    mApplyMask = bool(mEditParams.at<float>(0, APPLY_MASK_INDEX));
    mRemoveEdit = bool(mEditParams.at<float>(0, REMOVE_EDIT_INDEX));
    mRect.x = mEditParams.at<float>(0, RECT_X);
    mRect.y = mEditParams.at<float>(0, RECT_Y);
    mRect.width = mEditParams.at<float>(0, RECT_W);
    mRect.height = mEditParams.at<float>(0, RECT_H);
}

void MaskEdit::invertMask()
{
    Mat onesImage(mMask.rows, mMask.cols, CV_32FC3, Scalar(1, 1, 1));
    Mat temp = (onesImage - mMask);
    temp.copyTo(mMask);
    mMaskData.second = std::move(mMaskData.first.load(mMask));
}

void MaskEdit::initEditParams()
{

    for (auto i = 0; i < NO_EDIT_PARAMS; i++) mEditParams.at<float>(0, i) = 1.0;

    mEditParams.at<float>(0, RGB_GAMMA_INDEX) = 1.0;            // Gamma
    mEditParams.at<float>(0, RGB_INPUT_MIN_HIST_INDEX) = 0.0;   // Hist Min
    mEditParams.at<float>(0, RGB_OUTPUT_MIN_HIST_INDEX) = 0.0;  // Hist Min
    mEditParams.at<float>(0, RGB_INPUT_MAX_HIST_INDEX) = 1.0;   // Hist Max
    mEditParams.at<float>(0, RGB_OUTPUT_MAX_HIST_INDEX) = 1.0;  // Hist Max

    mEditParams.at<float>(0, R_GAMMA_INDEX) = 1.0;            // Gamma
    mEditParams.at<float>(0, R_INPUT_MIN_HIST_INDEX) = 0.0;   // Hist Min
    mEditParams.at<float>(0, R_OUTPUT_MIN_HIST_INDEX) = 0.0;  // Hist Min
    mEditParams.at<float>(0, R_INPUT_MAX_HIST_INDEX) = 1.0;   // Hist Max
    mEditParams.at<float>(0, R_OUTPUT_MAX_HIST_INDEX) = 1.0;  // Hist Max

    mEditParams.at<float>(0, G_GAMMA_INDEX) = 1.0;            // Gamma
    mEditParams.at<float>(0, G_INPUT_MIN_HIST_INDEX) = 0.0;   // Hist Min
    mEditParams.at<float>(0, G_OUTPUT_MIN_HIST_INDEX) = 0.0;  // Hist Min
    mEditParams.at<float>(0, G_INPUT_MAX_HIST_INDEX) = 1.0;   // Hist Max
    mEditParams.at<float>(0, G_OUTPUT_MAX_HIST_INDEX) = 1.0;  // Hist Max

    mEditParams.at<float>(0, B_GAMMA_INDEX) = 1.0;            // Gamma
    mEditParams.at<float>(0, B_INPUT_MIN_HIST_INDEX) = 0.0;   // Hist Min
    mEditParams.at<float>(0, B_OUTPUT_MIN_HIST_INDEX) = 0.0;  // Hist Min
    mEditParams.at<float>(0, B_INPUT_MAX_HIST_INDEX) = 1.0;   // Hist Max
    mEditParams.at<float>(0, B_OUTPUT_MAX_HIST_INDEX) = 1.0;  // Hist Max

    mEditParams.at<float>(0, EXPOSURE) = 0.0;

    mEditParams.at<float>(0, APPLY_MASK_INDEX) = 1.0;  // Apply Bool
    // mEditParams.at<float>(0, MASK_TYPE_INDEX) = 0.0;        // Mask Type
    mEditParams.at<float>(0, NO_REL_POS) = 0.0;

    mEditParams.at<float>(0, HUE_INDEX) = 0.0;
    mEditParams.at<float>(0, SATURATION_INDEX) = 0.0;
    mEditParams.at<float>(0, LIGHTNESSS_INDEX) = 0.0;

    mEditParams.at<float>(0, BRIGHTNESS_INDEX) = 0.0;
    mEditParams.at<float>(0, CONTRAST_INDEX) = 1.0;

    mEditParams.at<float>(0, BLUR_SIZE_INDEX) = 0.0;
    mEditParams.at<float>(0, BLUR_SIGMA_INDEX) = 2.0;

    for (auto i = 36; i < 47; i++) mEditParams.at<float>(0, i) = -1.0;
    // cout << "mEditParams init " << mEditParams << endl;
    mEditParams.copyTo(mOldParamsToggle);
}

void MaskEdit::setChannelIndexes(vector<int> indexes)
{
    int counter = 36;
    for (auto index : indexes) {
        cout << "counter " << counter << " index " << index << endl;
        mEditParams.at<float>(0, counter++) = float(index);
    }
    mChannelIndexes = indexes;
}

vector<int> MaskEdit::getChannelIndexes()
{
    vector<int> indexes;
    for (auto i = 36; i < 47; i++) {
        if (mEditParams.at<float>(0, i) == -1.0) {
            break;
        }
        indexes.push_back(int(mEditParams.at<float>(0, i)));
    }
    return indexes;
}

float MaskEdit::intersectionArea(JWH::Rectangle rect)
{
    Rect cvRect = JWH::convertToCVRect(rect, mMask);
    Rect intersectRect = cvRect & mRect;
    return intersectRect.area();
}

void MaskEdit::writeParameters(String filename)
{
    mEditParams.at<float>(0, RECT_X) = mRect.x;
    mEditParams.at<float>(0, RECT_Y) = mRect.y;
    mEditParams.at<float>(0, RECT_W) = mRect.width;
    mEditParams.at<float>(0, RECT_H) = mRect.height;
    mEditParams.at<float>(0, MATERIAL_ID_INDEX) = mMaterialIndex;

    cv::FileStorage file(filename, cv::FileStorage::WRITE);
    file.write("Params", mEditParams);

    /*    filename = filename.substr(0, filename.size() - 4);*/

    // for (auto i = 0; i < mRelativePatches.size(); i++) {
    // imwrite(filename + "_RelativePatch" + to_string(i) + ".png",
    // mRelativePatches[i]);

    // imwrite(filename + "_RelativePatchDelta" + to_string(i) + ".png",
    // mRelativePatchesDeltas[i]);
    /*}*/
}

void MaskEdit::transferParameters(vector<Mat> input_channels)
{
    mEditParams.copyTo(mOldParamsToggle);

    // mParamType = true;
    //// Target statistic
    //// Call parameter optimisation
    vector<Vector3d> neighboring_patches;
    vector<Vector3d> taget_distances;
    Mat input_channel;
    JWH::makeBeautyPassRGB(input_channel, input_channels);

    /*    imshow("Beauty", input_channel);*/
    // cout << input_channel << endl;
    /*waitKey();*/

    /*    int paramsIndex = REl_POS1_L;*/
    // for (auto i = 0; i < mRelativePatches.size(); i++) {
    // EditStat patchStats = getPatchStats(input_channel, mRelativePatches[i]);
    // Vector3f patch_mean =
    // JWH::eigenLab2RGB(patchStats.mean) * (1.0 / 255.0);
    // neighboring_patches.push_back(patch_mean.cast<double>());
    // float l = mEditParams.at<float>(0, paramsIndex);
    // float a = mEditParams.at<float>(0, paramsIndex + 1);
    // float b = mEditParams.at<float>(0, paramsIndex + 2);
    // Vector3f target_distance(l, a, b);
    //// target_distance = JWH::eigenLab2RGB(target_distance);
    // taget_distances.push_back(target_distance.cast<double>());
    // paramsIndex += 3;

    // cout << neighboring_patches[i].transpose() << endl;
    // cout << taget_distances[i].transpose() << endl;
    /*}*/

    PatchStat patchStats = getPatchStats(input_channel, mRelativePatches[0]);
    cout << "patchStats.mean " << patchStats.mean << endl;
    Vector3f target_mean = JWH::eigenLab2RGB(patchStats.mean) * (1.0 / 255.0);
    Vector3f target_stddev =
        JWH::eigenLab2RGB(patchStats.stddev) * (1.0 / 255.0);
    /*    Vector3f target_stddev =*/
    // JWH::eigenLab2RGB(patchStats.stddev) * (1.0 / 255.0);

    Eigen::Vector3d input_parameters(
        mEditParams.at<float>(0, R_GAMMA_INDEX),
        mEditParams.at<float>(0, R_INPUT_MIN_HIST_INDEX),
        mEditParams.at<float>(0, R_INPUT_MAX_HIST_INDEX));

    JWH::convertFromGamma<double>(&input_parameters(0));

    Mat singleChannelMask;
    mMask.convertTo(singleChannelMask, CV_8UC3, 255.0);
    cvtColor(singleChannelMask, singleChannelMask, COLOR_RGB2GRAY);

    cout << "Channels " << singleChannelMask.channels() << endl;

    int max_size = singleChannelMask.cols * singleChannelMask.rows;
    Eigen::MatrixXd input_patch(max_size, 3);
    int index_count = 0;
    for (auto i = 0; i < singleChannelMask.cols; i++) {
        for (auto j = 0; j < singleChannelMask.rows; j++) {
            if (singleChannelMask.at<uchar>(i, j) > 0.0 &&
                input_channel.at<Vec3f>(i, j)[0] > 0.01) {
                for (auto c = 0; c < 3; c++) {
                    float val = input_channel.at<Vec3f>(i, j)[c];
                    assert(!std::isnan(val) && std::isfinite(val));
                    input_patch(index_count, c) = val;
                }
                index_count++;
            }
        }
    }

    assert(index_count > 0);
    input_patch = input_patch.block(0, 0, index_count, 3);
    cout << "input_patch mean " << input_patch.colwise().mean() << endl;
    pair<Vector3d, Vector3d> stats =
        JWH::computeMeanStddev(input_patch, index_count);

    // imshow("mRelativePatches[0]", mRelativePatches[0]);
    // imshow("input_channel", input_channel);
    // imshow("singleChannelMask", singleChannelMask);
    /*waitKey();*/

    // target_stddev = stats.second;
    Mat targetMeanMat(300, 300, CV_32FC3,
                      Scalar(target_mean(0), target_mean(1), target_mean(2)));
    imshow("targetMean", targetMeanMat);

    Mat handPickedMeanMat(300, 300, CV_32FC3,
                          Scalar(0.844656, 0.845109, 0.866639));
    imshow("handPickedMean", handPickedMeanMat);

    Eigen::Vector3d output_parameters =
        ParameterOptimisation::run(target_mean.cast<double>(), stats.second,
                                   input_patch, input_parameters);

    JWH::convertToGamma<double>(&output_parameters(0));

    mEditParams.at<float>(0, R_GAMMA_INDEX) = output_parameters(0);
    mEditParams.at<float>(0, R_INPUT_MIN_HIST_INDEX) = output_parameters(1);
    mEditParams.at<float>(0, R_INPUT_MAX_HIST_INDEX) = output_parameters(2);

    cout << mOldParamsToggle << endl;
    cout << mEditParams << endl;
}

pair<PatchStat, PatchStat> MaskEdit::getEditStats(const Mat& input_channel)
{

    Eigen::Vector3f input_parameters(
        mEditParams.at<float>(0, R_GAMMA_INDEX),
        mEditParams.at<float>(0, R_INPUT_MIN_HIST_INDEX),
        mEditParams.at<float>(0, R_INPUT_MAX_HIST_INDEX));

    JWH::convertFromGamma<float>(&input_parameters(0));

    Mat singleChannelMask;
    mMask.convertTo(singleChannelMask, CV_8UC3, 255.0);

    auto no_pixels = 0;

    int size = singleChannelMask.cols * singleChannelMask.rows;
    assert(size > 0);

    Eigen::MatrixXf editedPixels(size, 3);
    Eigen::MatrixXf inputPixels(size, 3);

    for (auto i = 0; i < singleChannelMask.cols; i++) {
        for (auto j = 0; j < singleChannelMask.rows; j++) {
            if (singleChannelMask.at<Vec3b>(i, j)[0] > 0 &&
                input_channel.at<Vec3f>(i, j)[0] > 0.0) {
                // cout << "Add Pixel " << endl;
                Eigen::Matrix<float, 3, 1> vec(
                    input_channel.at<Vec3f>(i, j)[0],
                    input_channel.at<Vec3f>(i, j)[1],
                    input_channel.at<Vec3f>(i, j)[2]);

                Eigen::Matrix<float, 3, 1> new_pixel =
                    ParameterOptimisation::applyEdit<float>(vec,
                                                            input_parameters);
                inputPixels.row(no_pixels) = vec;
                editedPixels.row(no_pixels) = new_pixel;
                no_pixels++;
            }
        }
    }

    PatchStat inputStats;
    PatchStat editedStats;

    if (no_pixels == 0) {
        cout << "No pixels found to compute edit mean " << endl;
        return make_pair(inputStats, editedStats);
    }

    pair<Vector3f, Vector3f> inputMeanStddev =
        JWH::computeMeanStddev(inputPixels, no_pixels);

    inputStats.mean = inputMeanStddev.first;
    inputStats.stddev = inputMeanStddev.second;

    pair<Vector3f, Vector3f> editedMeanStddev =
        JWH::computeMeanStddev(editedPixels, no_pixels);

    editedStats.mean = editedMeanStddev.first;
    editedStats.stddev = editedMeanStddev.second;

    editedStats.mean = JWH::eigenRGB2Lab(editedStats.mean);
    editedStats.stddev = JWH::eigenRGB2Lab(editedStats.stddev);

    inputStats.mean = JWH::eigenRGB2Lab(inputStats.mean);
    inputStats.stddev = JWH::eigenRGB2Lab(inputStats.stddev);

    return make_pair(inputStats, editedStats);
}

float MaskEdit::setRGBGamma(float gamma)
{
    JWH::convertToGamma<float>(&gamma);
    mEditParams.at<float>(0, RGB_GAMMA_INDEX) = gamma;
    mParameterCallback();
    return gamma;
}
float MaskEdit::getRGBGamma()
{
    float g = mEditParams.at<float>(0, RGB_GAMMA_INDEX);
    JWH::convertFromGamma<float>(&g);
    return g;
}

void MaskEdit::setRGBHistMin(float value)
{
    mEditParams.at<float>(0, RGB_INPUT_MIN_HIST_INDEX) = value;
    mParameterCallback();
}

float MaskEdit::getRGBHistMin()
{
    return mEditParams.at<float>(0, RGB_INPUT_MIN_HIST_INDEX);
}

void MaskEdit::setRGBHistMax(float value)
{
    mEditParams.at<float>(0, RGB_INPUT_MAX_HIST_INDEX) = value;
    mParameterCallback();
}

float MaskEdit::getRGBHistMax()
{
    return mEditParams.at<float>(0, RGB_INPUT_MAX_HIST_INDEX);
}

void MaskEdit::setRGBHistOutMin(float value)
{
    mEditParams.at<float>(0, RGB_OUTPUT_MIN_HIST_INDEX) = value;
    mParameterCallback();
}
float MaskEdit::getRGBHistOutMin()
{
    return mEditParams.at<float>(0, RGB_OUTPUT_MIN_HIST_INDEX);
}

void MaskEdit::setRGBHistOutMax(float value)
{
    mEditParams.at<float>(0, RGB_OUTPUT_MAX_HIST_INDEX) = value;
    mParameterCallback();
}
float MaskEdit::getRGBHistOutMax()
{
    return mEditParams.at<float>(0, RGB_OUTPUT_MAX_HIST_INDEX);
}

float MaskEdit::setRGamma(float gamma)
{
    JWH::convertToGamma<float>(&gamma);
    mEditParams.at<float>(0, R_GAMMA_INDEX) = gamma;
    mParameterCallback();
    return gamma;
}

float MaskEdit::getRGamma()
{
    float g = mEditParams.at<float>(0, R_GAMMA_INDEX);
    JWH::convertFromGamma<float>(&g);
    return g;
}

void MaskEdit::setRHistMin(float value)
{
    mEditParams.at<float>(0, R_INPUT_MIN_HIST_INDEX) = value;
    mParameterCallback();
}

float MaskEdit::getRHistMin()
{
    return mEditParams.at<float>(0, R_INPUT_MIN_HIST_INDEX);
}

void MaskEdit::setRHistMax(float value)
{
    mEditParams.at<float>(0, R_INPUT_MAX_HIST_INDEX) = value;
    mParameterCallback();
}

float MaskEdit::getRHistMax()
{
    return mEditParams.at<float>(0, R_INPUT_MAX_HIST_INDEX);
}

void MaskEdit::setRHistOutMin(float value)
{
    mEditParams.at<float>(0, R_OUTPUT_MIN_HIST_INDEX) = value;
    mParameterCallback();
}
float MaskEdit::getRHistOutMin()
{
    return mEditParams.at<float>(0, R_OUTPUT_MIN_HIST_INDEX);
}

void MaskEdit::setRHistOutMax(float value)
{
    mEditParams.at<float>(0, R_OUTPUT_MAX_HIST_INDEX) = value;
    mParameterCallback();
}
float MaskEdit::getRHistOutMax()
{
    return mEditParams.at<float>(0, R_OUTPUT_MAX_HIST_INDEX);
}

float MaskEdit::setGGamma(float gamma)
{
    JWH::convertToGamma<float>(&gamma);
    mEditParams.at<float>(0, G_GAMMA_INDEX) = gamma;
    mParameterCallback();
    return gamma;
}
float MaskEdit::getGGamma()
{
    float g = mEditParams.at<float>(0, G_GAMMA_INDEX);
    JWH::convertFromGamma<float>(&g);
    return g;
}

void MaskEdit::setGHistMin(float value)
{
    mEditParams.at<float>(0, G_INPUT_MIN_HIST_INDEX) = value;
    mParameterCallback();
}

float MaskEdit::getGHistMin()
{
    return mEditParams.at<float>(0, G_INPUT_MIN_HIST_INDEX);
}

void MaskEdit::setGHistMax(float value)
{
    mEditParams.at<float>(0, G_INPUT_MAX_HIST_INDEX) = value;
    mParameterCallback();
}

float MaskEdit::getGHistMax()
{
    return mEditParams.at<float>(0, G_INPUT_MAX_HIST_INDEX);
}

void MaskEdit::setGHistOutMin(float value)
{
    mEditParams.at<float>(0, G_OUTPUT_MIN_HIST_INDEX) = value;
    mParameterCallback();
}
float MaskEdit::getGHistOutMin()
{
    return mEditParams.at<float>(0, G_OUTPUT_MIN_HIST_INDEX);
}

void MaskEdit::setGHistOutMax(float value)
{
    mEditParams.at<float>(0, G_OUTPUT_MAX_HIST_INDEX) = value;
    mParameterCallback();
}
float MaskEdit::getGHistOutMax()
{
    return mEditParams.at<float>(0, G_OUTPUT_MAX_HIST_INDEX);
}

float MaskEdit::setBGamma(float gamma)
{
    JWH::convertToGamma<float>(&gamma);
    mEditParams.at<float>(0, B_GAMMA_INDEX) = gamma;
    mParameterCallback();
    return gamma;
}
float MaskEdit::getBGamma()
{
    float g = mEditParams.at<float>(0, B_GAMMA_INDEX);
    JWH::convertFromGamma<float>(&g);
    return g;
}

void MaskEdit::setBHistMin(float value)
{
    mEditParams.at<float>(0, B_INPUT_MIN_HIST_INDEX) = value;
    mParameterCallback();
}

float MaskEdit::getBHistMin()
{
    return mEditParams.at<float>(0, B_INPUT_MIN_HIST_INDEX);
}

void MaskEdit::setBHistMax(float value)
{
    mEditParams.at<float>(0, B_INPUT_MAX_HIST_INDEX) = value;
    mParameterCallback();
}

float MaskEdit::getBHistMax()
{
    return mEditParams.at<float>(0, B_INPUT_MAX_HIST_INDEX);
}

void MaskEdit::setBHistOutMin(float value)
{
    mEditParams.at<float>(0, B_OUTPUT_MIN_HIST_INDEX) = value;
    mParameterCallback();
}
float MaskEdit::getBHistOutMin()
{
    return mEditParams.at<float>(0, B_OUTPUT_MIN_HIST_INDEX);
}

void MaskEdit::setBHistOutMax(float value)
{
    mEditParams.at<float>(0, B_OUTPUT_MAX_HIST_INDEX) = value;
    mParameterCallback();
}
float MaskEdit::getBHistOutMax()
{
    return mEditParams.at<float>(0, B_OUTPUT_MAX_HIST_INDEX);
}

void MaskEdit::setHue(float value)
{
    mEditParams.at<float>(0, HUE_INDEX) = ((value)-0.5);
    mParameterCallback();
}

float MaskEdit::getHue()
{
    float val = mEditParams.at<float>(0, HUE_INDEX) + 0.5;
    return val;
}

void MaskEdit::setSaturation(float value)
{
    mEditParams.at<float>(0, SATURATION_INDEX) = value - 0.5;
    mParameterCallback();
}

float MaskEdit::getSaturation()
{
    return mEditParams.at<float>(0, SATURATION_INDEX) + 0.5;
}

void MaskEdit::setLightness(float value)
{
    mEditParams.at<float>(0, LIGHTNESSS_INDEX) = value - 0.5;
    mParameterCallback();
}

float MaskEdit::getLightness()
{
    return mEditParams.at<float>(0, LIGHTNESSS_INDEX) + 0.5;
}

void MaskEdit::setBrightness(float value)
{
    mEditParams.at<float>(0, BRIGHTNESS_INDEX) = (value - 0.5);
    mParameterCallback();
}

float MaskEdit::getBrightness()
{
    return mEditParams.at<float>(0, BRIGHTNESS_INDEX) + 0.5;
}

void MaskEdit::setContrast(float value)
{
    if (value == 0.5) {
        mEditParams.at<float>(0, CONTRAST_INDEX) = 1.0;
    }
    else if (value > 0.5) {
        value -= 0.5;
        value /= 0.5;
        mEditParams.at<float>(0, CONTRAST_INDEX) = 1.0 + (value * 5.0);
    }
    else {
        value /= 0.5;
        mEditParams.at<float>(0, CONTRAST_INDEX) = value;
    }
    mParameterCallback();
}

float MaskEdit::getContrast()
{
    float value = 0.5;
    if (value > 1.0) {
        value = (mEditParams.at<float>(0, CONTRAST_INDEX) / 5.0) - 1.0;
        value *= 0.5;
        value += 0.5;
    }
    else if (value < 1.0) {
        value = mEditParams.at<float>(0, CONTRAST_INDEX) * 0.5;
    }
    return value;
}

void MaskEdit::setExposure(float value)
{
    mEditParams.at<float>(0, EXPOSURE) =
        ((value * 2.0) - 1.0) * 10.0;  //(value * 2.0 - 1.0) * 10.0
    mParameterCallback();
}

float MaskEdit::getExposure()
{
    float val = ((mEditParams.at<float>(0, EXPOSURE) - -10.0) / 20.0);
    return val;
}

void MaskEdit::setBlurSize(int value)
{
    mEditParams.at<float>(0, BLUR_SIZE_INDEX) = (float) value;
    mParameterCallback();
}

int MaskEdit::getBlurSize(){
  return int(mEditParams.at<float>(0, BLUR_SIZE_INDEX));
}

void MaskEdit::setBlurSigma(float value)
{
  mEditParams.at<float>(0, BLUR_SIGMA_INDEX) = value;
  mParameterCallback();
}

float MaskEdit::getBlurSigma()
{
  return mEditParams.at<float>(0, BLUR_SIGMA_INDEX);
}

void MaskEdit::applyMaskToLayer(bool val)
{
    mEditParams.at<float>(0, APPLY_MASK_INDEX) = float(val);
    mParameterCallback();
}

bool MaskEdit::applyMask()
{
    return mEditParams.at<float>(0, APPLY_MASK_INDEX) = float(true);
    mParameterCallback();
}

/*void MaskEdit::computeEditRatio(Mat channel)*/
//{
// pair<PatchStat, PatchStat> edit_stats = getPatchStats(channel);

// PatchStat neighbourStats =
// EditEmbedding::computeNeightbourStatistics(channel, mMask, mMaterialId);

// cout << "Input " << edit_stats.first.mean.transpose() << " "
//<< edit_stats.first.stddev.transpose() << endl;

// cout << "Output " << edit_stats.second.mean.transpose() << " "
//<< edit_stats.second.stddev.transpose() << endl;

// cout << "Neighbour " << neighbourStats.mean.transpose() << " "
//<< neighbourStats.stddev.transpose() << endl;

// Eigen::Vector3f delta = edit_stats.second.mean - neighbourStats.mean;

// Eigen::Vector3f ratio = delta.cwiseQuotient(neighbourStats.stddev);
// Eigen::Vector3f stddev_ratio =
// edit_stats.first.stddev.cwiseQuotient(edit_stats.second.stddev);

// for (auto i = 0; i < 3; i++) {
// if (delta(i) < 0.0001) {
// ratio(i) = 0;
// stddev_ratio(i) = 0;
//}
//}

// cout << "Ratio " << ratio.transpose() << endl;
// mEditParams.at<float>(0, RATIO_L) = ratio(0);
// mEditParams.at<float>(0, RATIO_A) = ratio(1);
// mEditParams.at<float>(0, RATIO_B) = ratio(2);

// mEditParams.at<float>(0, RATIO_STDDEV_L) = stddev_ratio(0);
// mEditParams.at<float>(0, RATIO_STDDEV_A) = stddev_ratio(1);
// mEditParams.at<float>(0, RATIO_STDDEV_B) = stddev_ratio(2);
/*}*/

PatchStat getPatchStats(Mat channel, Mat patch)
{
    int max_pixels = channel.rows * channel.cols;
    MatrixXf allPixels(max_pixels, 3);

    Mat singleChannelMask;
    patch.convertTo(singleChannelMask, CV_8UC3, 255.0);

    int pixelCount = 0;
    for (auto i = 0; i < channel.rows; i++) {
        for (auto j = 0; j < channel.cols; j++) {
            if (singleChannelMask.at<Vec3b>(i, j)[0] > 10 &&
                channel.at<Vec3f>(i, j)[0] > 0) {

                Vec3f p = channel.at<Vec3f>(i, j);
                allPixels.row(pixelCount) = Vector3f(p[0], p[1], p[2]);
                // channel.at<Vec3f>(i, j) = Vec3f(1, 0, 0);
                pixelCount++;
            }
        }
    }

    // imshow("channel", channel);

    pair<Vector3f, Vector3f> patchdMeanStddev =
        JWH::computeMeanStddev(allPixels, pixelCount);

    PatchStat patchStat;

    patchStat.mean = JWH::eigenRGB2Lab(patchdMeanStddev.first);
    patchStat.stddev = JWH::eigenRGB2Lab(patchdMeanStddev.second);

    return patchStat;
}

void MaskEdit::computeRelativePositioning(Mat beforeEdit, Mat afterEdit,
                                          vector<Mat> patches)
{
    pair<PatchStat, PatchStat> beforeEditStat = getEditStats(beforeEdit);
    pair<PatchStat, PatchStat> afterEditStat = getEditStats(afterEdit);

    mEditParams.at<float>(0, NO_REL_POS) = patches.size();

    mRelativePatches.clear();
    mRelativePatchesDeltas.clear();

    int paramsIndex = REl_POS1_L;
    for (auto i = 0; i < patches.size(); i++) {
        PatchStat patchStats = getPatchStats(beforeEdit, patches[i]);

        /*        Mat singleChannelMask;*/
        // patches[i].convertTo(singleChannelMask, CV_8UC3, 255.0);
        /*cvtColor(singleChannelMask, singleChannelMask, COLOR_RGB2GRAY);*/

        Mat maskApplied;
        beforeEdit.copyTo(maskApplied, patches[i]);
        maskApplied.convertTo(maskApplied, CV_8UC3, 255.0);
        mRelativePatchesDeltas.push_back(maskApplied);
        /*        imshow("maskApplied", maskApplied);*/
        /*waitKey();*/

        mRelativePatches.push_back(patches[i]);

        Eigen::Vector3f beforeDelta =
            JWH::eigenLab2RGB(beforeEditStat.first.mean) -
            JWH::eigenLab2RGB(patchStats.mean);

        Eigen::Vector3f afterDelta =
            JWH::eigenLab2RGB(afterEditStat.first.mean) -
            JWH::eigenLab2RGB(patchStats.mean);

        cout << "beforeEditStat.first.mean "
             << beforeEditStat.first.mean.transpose() << endl;
        cout << "patchStats.mean " << patchStats.mean.transpose() << endl;
        cout << "beforeDelta " << beforeDelta.transpose() << endl;
        cout << "afterDelta " << afterDelta.transpose() << endl;

        afterDelta = afterDelta.cwiseQuotient(beforeDelta);
        beforeDelta = beforeDelta.cwiseQuotient(beforeDelta);

        afterDelta = afterDelta.array() - 1.0f;
        cout << " afterDelta" << afterDelta.transpose() << endl;

        mEditParams.at<float>(0, paramsIndex) = afterDelta(0);
        mEditParams.at<float>(0, paramsIndex + 1) = afterDelta(1);
        mEditParams.at<float>(0, paramsIndex + 2) = afterDelta(2);
        paramsIndex += 3;
    }
}

/*     bool setMaskType(float value)*/
//{
// return mEditParams.at<float>(0, MASK_TYPE_INDEX) = value;
/*}*/
void MaskEdit::setRemoveEdit(bool val)
{
    mRemoveEdit = val;
    mEditParams.at<float>(0, REMOVE_EDIT_INDEX) = float(val);
    mParameterCallback();
}
bool MaskEdit::getRemoveEdit() { return mRemoveEdit; }

JWH::Rectangle MaskEdit::getRectangle()
{
    return JWH::Rectangle(mRect.x, mRect.y, mRect.width, mRect.height);
}

void MaskEdit::toggleMaskParams()
{
    // cout << mOldParamsToggle << endl;

    Mat temp;
    mEditParams.copyTo(temp);
    mOldParamsToggle.copyTo(mEditParams);
    temp.copyTo(mOldParamsToggle);
    if (mParamType)
        cout << "Transfered Params " << endl;
    else
        cout << "Params from view 1 " << endl;

    mParamType = !mParamType;
}

void MaskEdit::copySegmentationToMask(Mat segmentation, Mat initialMask)
{
    Mat floatImage(segmentation.rows, segmentation.cols, CV_32FC3,
                   Scalar(0, 0, 0));
    for (auto i = 0; i < segmentation.rows; i++) {
        for (auto j = 0; j < segmentation.cols; j++) {
            if (segmentation.at<char>(i, j) == GC_FGD ||
                segmentation.at<char>(i, j) == GC_PR_FGD) {

                floatImage.at<Vec3f>(i, j) = Vec3f(1, 1, 1);
            }
            else {
                floatImage.at<Vec3f>(i, j) = Vec3f(0, 0, 0);
            }
        }
    }
    floatImage.copyTo(mMask);

    mMaskData.second = std::move(mMaskData.first.load(mMask));
}

void MaskEdit::drawMask(Eigen::Vector2f p1, Eigen::Vector2f p2, int size,
                        float intensity)
{
    if (p1 == p2) {
        circle(mMask, Point(int(p1(0)), int(p1(1))), size / 2,
               Scalar(intensity, intensity, intensity), -1);
    }
    else {
        line(mMask, Point(int(p1(0)), int(p1(1))),
             Point(int(p2(0)), int(p2(1))),
             Scalar(intensity, intensity, intensity), size, LINE_AA);
    }
    // mMaskData.second = std::move(mMaskData.first.load(mMask));
    mMaskData.first.update(mMask);
}

// Segmentation image comes as 8bit single channel, we need a normal 32bit
// mask
Mat segmentationToFloatImage(Mat segmentation)
{
    cout << "segmentationToFloatImage " << endl;
    // Already float image, no need to segment
    if (segmentation.type() == 21) {
        cout << "segmentation.type() == 21 " << endl;
        return segmentation;
    }

    Mat floatImage(segmentation.rows, segmentation.cols, CV_32FC3,
                   Scalar(0, 0, 0));
    for (auto i = 0; i < segmentation.rows; i++) {
        for (auto j = 0; j < segmentation.cols; j++) {
            if (segmentation.at<char>(i, j) == GC_FGD ||
                segmentation.at<char>(i, j) == GC_PR_FGD) {

                /*                if (i > 1) floatImage.at<Vec3f>(i - 1, j) =
                 * Vec3f(0, 0, 0);*/
                // if (i < (segmentation.rows - 1))
                // floatImage.at<Vec3f>(i + 1, j) = Vec3f(0, 0, 0);

                // if (j > 1) floatImage.at<Vec3f>(i, j + 1) = Vec3f(0, 0, 0);
                // if (j < (segmentation.cols - 1))
                /*floatImage.at<Vec3f>(i, j + 1) = Vec3f(0, 0, 0);*/

                floatImage.at<Vec3f>(i, j) = Vec3f(1, 1, 1);
            }
        }
    }
    return floatImage;
}
