#ifndef JWH_UTIL_H
#define JWH_UTIL_H

#include <Eigen/Core>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;
using namespace Eigen;

namespace JWH {

inline string readFile(const char *filepath)
{
    ifstream myFile(filepath);
    string content((istreambuf_iterator<char>(myFile)),
                   istreambuf_iterator<char>());
    return content;
}

struct Rectangle {
    Eigen::Vector4f rect;
    Rectangle() : rect(0, 0, 0, 0) {}
    Rectangle(int x, int y, int w, int h) : rect(x, y, w, h) {}
    int width() { return rect[2]; }
    int height() { return rect[3]; }
    int x() { return rect[0]; }
    int y() { return rect[1]; }
    Eigen::Vector4f getRect() { return rect; }
    Eigen::Vector2f getPosVector() { return Eigen::Vector2f(x(), y()); }
    Eigen::Vector2f getWHVector() { return Eigen::Vector2f(width(), height()); }
    void setPoint(Eigen::Vector2f pos)
    {
        rect[0] = pos[0];
        rect[1] = pos[1];
    }
    void setWidthHeight(Eigen::Vector2f diff)
    {
        rect[2] = diff[0] - rect[0];
        rect[3] = diff[1] - rect[1];
    }
    void clear() { rect = Eigen::Vector4f(0, 0, 0, 0); }
    bool valid() { return (rect[2] != 0 && rect[3] != 3); }
    float area() { return width() * height(); }
    bool same(Rectangle rect2)
    {
        return (rect2.getRect() - rect).norm() == 0.0f;
    }
    bool inside(int _x, int _y)
    {
        if (_x < x() || _x > x() + width() || _y < y() || _y > y() + height())
            return false;

        return true;
    }
};

inline cv::Rect convertToCVRect(JWH::Rectangle rectangle, cv::Mat image)
{
    cv::Rect rect = cv::Rect(rectangle.x(), rectangle.y(), rectangle.width(),
                             rectangle.height());
    assert(rect.area() > 0);

    if (rect.x < 0) rect.x = 0;
    if (rect.y < 0) rect.y = 0;

    int end_x = rect.x + rect.width;
    int end_y = rect.y + rect.height;

    if (end_x > image.cols) rect.width = rect.width - (end_x - image.cols);

    if (end_y > image.rows) rect.height = rect.height - (end_y - image.rows);

    return rect;
}

template <typename T>
T convertToGamma(const T gamma)
{
    //   cout << gamma << " ";
    T gammaOut = gamma;
    if (gammaOut >= T(0.5)) {
        gammaOut = (T(1.0) - ((gammaOut - T(0.5)) / T(0.5)));
    }
    else {
        gammaOut = T(1.0) - gamma / T(0.5);
        gammaOut = T(1.0) + gammaOut * T(3.99);
    }
    // cout << gammaOut << endl;
    return gammaOut;
}

template <typename T>
void convertToGamma(T *gamma)
{
    // cout << gamma[0] << " ";
    if (gamma[0] >= T(0.5)) {
        gamma[0] = (T(1.0) - ((gamma[0] - T(0.5)) / T(0.5)));
    }
    else {
        gamma[0] = T(1.0) - gamma[0] / T(0.5);
        gamma[0] = T(1.0) + gamma[0] * T(3.99);
    }
    // cout << gamma[0] << endl;
}

template <typename T>
void convertFromGamma(T *gamma)
{

    // cout << "convertFromGamma " << gamma[0] << " ";
    if (gamma[0] <= T(1.0)) {
        gamma[0] = T(1.0) - (gamma[0] / T(2.0));
    }
    else {
        gamma[0] = (T(1.0) - gamma[0]) / T(3.99);
        // cout << gamma[0] << "  " << endl;
        gamma[0] = T(0.5) + gamma[0] * T(0.5);
    }
    // cout << gamma[0] << endl;
}

inline cv::Vec3f eigenLab2RGBCV(Eigen::Vector3f lab)
{
    cv::Mat labConversion(1, 1, CV_32FC3);
    labConversion.at<cv::Vec3f>(0, 0) = cv::Vec3f(lab(0), lab(1), lab(2));
    cvtColor(labConversion, labConversion, CV_Lab2RGB);
    labConversion.at<cv::Vec3f>(0, 0) *= 255.0;
    return labConversion.at<cv::Vec3f>(0, 0);
}

inline Eigen::Vector3f eigenLab2RGB(Eigen::Vector3f lab)
{
    cv::Vec3f rgb = eigenLab2RGBCV(lab);
    Eigen::Vector3f eigenRGBNew(fabs(rgb[0]), fabs(rgb[1]), fabs(rgb[2]));
    return eigenRGBNew;
}

inline cv::Vec3f eigenRGB2LabCV(Eigen::Vector3f rgb)
{
    cv::Mat labConversion(1, 1, CV_32FC3);
    labConversion.at<cv::Vec3f>(0, 0) = cv::Vec3f(rgb(0), rgb(1), rgb(2));
    cvtColor(labConversion, labConversion, CV_BGR2Lab);
    cv::Vec3f lab = labConversion.at<cv::Vec3f>(0, 0);
    return lab;
}

inline Eigen::Vector3f eigenRGB2Lab(Eigen::Vector3f rgb)
{
    cv::Vec3f lab = eigenRGB2LabCV(rgb);
    Eigen::Vector3f eigenLabNew(lab[0], lab[1], lab[2]);
    return eigenLabNew;
}

template <typename T>
inline pair<Matrix<T, 3, 1>, Matrix<T, 3, 1>> computeMeanStddev(
    Matrix<T, Dynamic, Dynamic> allPixels, int count)
{

    Matrix<T, 3, 1> mean = Matrix<T, 3, 1>::Zero();
    Matrix<T, 3, 1> stddev = Matrix<T, 3, 1>::Zero();
    if (count > 0) {
        Matrix<T, Dynamic, Dynamic> nonBlackPixels =
            allPixels.block(0, 0, count, 3);

        nonBlackPixels = nonBlackPixels.array().min(1.0);
        mean = nonBlackPixels.colwise().mean();
        nonBlackPixels = nonBlackPixels.rowwise() - mean.transpose();

        nonBlackPixels = nonBlackPixels.cwiseAbs();
        stddev = nonBlackPixels.colwise().mean();
    }
    auto stats = make_pair(mean, stddev);
    return stats;
}

template <typename T>
inline bool median(const Matrix<T, 1, Dynamic> V, T &m)
{
    using namespace std;
    if (V.size() == 0) {
        return false;
    }
    vector<T> vV;

    for (int i = 0; i < V.rows(); i++) {
        vV.push_back(V(i));
    }
    // http://stackoverflow.com/a/1719155/148668
    size_t n = vV.size() / 2;
    nth_element(vV.begin(), vV.begin() + n, vV.end());
    if (vV.size() % 2 == 0) {
        nth_element(vV.begin(), vV.begin() + n - 1, vV.end());
        m = 0.5 * (vV[n] + vV[n - 1]);
    }
    else {
        m = vV[n];
    }
    return true;
}

template <typename T>
inline pair<Matrix<T, 3, 1>, Matrix<T, 3, 1>> computeMedianStddev(
    Matrix<T, Dynamic, Dynamic> allPixels, int count)
{

    Matrix<T, 3, 1> mean = Matrix<T, 3, 1>::Zero();
    Matrix<T, 3, 1> stddev = Matrix<T, 3, 1>::Zero();
    if (count > 0) {
        Matrix<T, Dynamic, Dynamic> nonBlackPixels =
            allPixels.block(0, 0, count, 3);

        nonBlackPixels = nonBlackPixels.array().min(1.0);
        // mean = nonBlackPixels.colwise().mean();
        median<T>(nonBlackPixels.col(0), mean(0));
        median<T>(nonBlackPixels.col(1), mean(1));
        median<T>(nonBlackPixels.col(2), mean(2));

        nonBlackPixels = nonBlackPixels.rowwise() - mean.transpose();

        nonBlackPixels = nonBlackPixels.cwiseAbs();
        stddev = nonBlackPixels.colwise().mean();
    }
    auto stats = make_pair(mean, stddev);
    return stats;
}

inline void makeBeautyPassRGB(cv::Mat &beautyPass, vector<cv::Mat> imageMats)
{
    beautyPass.create(imageMats[0].rows, imageMats[0].cols, CV_32FC3);
    for (auto i = 1; i < imageMats.size(); i++) {
        beautyPass += imageMats[i];
    }
}

inline void makeBeautyPassLab(cv::Mat &beautyPass, vector<cv::Mat> imageMats)
{
    beautyPass.create(imageMats[0].rows, imageMats[0].cols, CV_32FC3);
    for (auto i = 1; i < imageMats.size(); i++) {
        cv::Mat labColour;
        cvtColor(imageMats[i], labColour, CV_BGR2Lab);
        beautyPass += labColour;
    }
}

/*Vector3f rgb2lab(Vector3f rgb)*/
//{
// float x, y, z;
// float r = rgb(2);
// float g = rgb(1);
// float b = rgb(0);

// r = (r > 0.04045) ? pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
// g = (g > 0.04045) ? pow((g + 0.055) / 1.055, 2.4) : g / 12.92;
// b = (b > 0.04045) ? pow((b + 0.055) / 1.055, 2.4) : b / 12.92;

// x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047;
// y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000;
// z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883;

// x = (x > 0.008856) ? pow(x, 1 / 3) : (7.787 * x) + 16 / 116;
// y = (y > 0.008856) ? pow(y, 1 / 3) : (7.787 * y) + 16 / 116;
// z = (z > 0.008856) ? pow(z, 1 / 3) : (7.787 * z) + 16 / 116;

// return Vector3f((116 * y) - 16, 500 * (x - y), 200 * (y - z));
/*}*/
}

#endif
