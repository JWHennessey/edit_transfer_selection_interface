#include "ChannelStat.h"
#include <iostream>
#include <list>
#include <queue>
#define TAPKEE_EIGEN_INCLUDE_FILE <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <ctime>
#include <iomanip>
#include <sstream>
#include "MaskEdit.h"
#include "tapkee/tapkee.hpp"

ChannelStat::ChannelStat(vector<Vector3f> means, vector<Vector3f> stddevs,
                         String filename)
    : mMeans(means), mStddevs(stddevs), mFilename(filename)
{
    for (auto i = 0; i < means.size(); i++) {
        cout << mFilename << endl;
        cout << "mMeans " << mMeans[i].transpose() << endl;
        cout << "mStddevs " << mStddevs[i].transpose() << endl;
    }
}

ChannelStat ChannelStat::computeChannelStatistics(String filename,
                                                  Mat channel_input,
                                                  Mat material_id,
                                                  Mat edit_mask)
{

    Mat mask_inverted;
    cvtColor(edit_mask, mask_inverted, COLOR_BGR2GRAY);
    mask_inverted.convertTo(mask_inverted, CV_8UC1, 255.0);

    Mat onesImage(mask_inverted.rows, mask_inverted.cols, CV_8UC1, Scalar(255));
    Mat temp = (onesImage - mask_inverted);
    temp.copyTo(mask_inverted);

    Mat masked_channel_temp;
    channel_input.copyTo(masked_channel_temp, material_id);

    Mat masked_channel;
    masked_channel_temp.copyTo(masked_channel, mask_inverted);

    masked_channel.convertTo(masked_channel, CV_8UC1, 255.0);
    cvtColor(masked_channel, masked_channel, COLOR_BGR2GRAY);

    Mat binary_channel;
    threshold(masked_channel, binary_channel, 1, 255, THRESH_BINARY);
    Mat connected_components;
    // imshow("binary_channel", binary_channel);

    Mat stats;
    Mat centroids;
    int no_cc = connectedComponentsWithStats(
        binary_channel, connected_components, stats, centroids);
    int size = channel_input.rows * channel_input.cols;

    Mat component_image(channel_input.rows, channel_input.cols, CV_32FC3,
                        Scalar(0));

    Mat grey;
    channel_input.convertTo(grey, CV_8UC3, 255.0);
    cvtColor(grey, grey, COLOR_BGR2GRAY);

    Mat channel;  // = channel_input;
    cvtColor(channel_input, channel, CV_BGR2Lab);
    // imshow("channel_input", channel_input);

    vector<Vector3f> means;
    vector<Vector3f> stddevs;
    /*    vector<Mat> centroids;*/
    /*vector<Mat> stats;*/
    for (auto cc = 1; cc < no_cc; cc++) {
        MatrixXf pixels(size, 3);
        int pixel_count = 0;
        for (auto i = 0; i < channel_input.rows; i++) {
            for (auto j = 0; j < channel_input.cols; j++) {
                if (connected_components.at<int>(i, j) == cc &&
                    grey.at<uchar>(i, j) > 3) {
                    Vec3f cvP = channel.at<Vec3f>(i, j);
                    Vector3f p(cvP(0), cvP(1), cvP(2));
                    pixels.row(pixel_count) = p;
                    pixel_count++;
                }
            }
        }
        // cout << pixel_count << endl;
        if (pixel_count > 100) {
            auto stats = JWH::computeMeanStddev(pixels, pixel_count);
            means.push_back(stats.first);
            stddevs.push_back(stats.second);
            // centroids.push_back(Vector2i());
            for (auto i = 0; i < channel_input.rows; i++) {
                for (auto j = 0; j < channel_input.cols; j++) {
                    if (connected_components.at<int>(i, j) == cc) {
                        Vec3f vec = JWH::eigenLab2RGBCV(stats.first);
                        component_image.at<Vec3f>(i, j) =
                            Vec3f(vec[2], vec[1], vec[0]);
                    }
                }
            }
        }
    }

    // filename
    auto found = filename.find("_.");
    filename = filename.substr(found + 2);
    filename = filename.substr(0, filename.size() - 4);

    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%d-%m-%Y_%H-%M-%S");
    auto str = oss.str();

    imwrite(filename + str + ".png", component_image);
    /*    imshow("component_image", component_image);*/
    /*waitKey();*/
    ChannelStat cs(means, stddevs, filename);
    return cs;
    /*    int size = channel_input.rows * channel_input.cols;*/

    // assert(size > 0);
    // cout << size << endl;
    // MatrixXf shadows(size, 3);
    // MatrixXf midtones(size, 3);
    // MatrixXf highlights(size, 3);

    // Vector3i count = Vector3i::Zero();

    // Mat grey;
    // channel_input.convertTo(grey, CV_8UC3, 255.0);
    // cvtColor(grey, grey, COLOR_BGR2GRAY);

    // Mat channel;  // = channel_input;
    // cvtColor(channel_input, channel, CV_BGR2Lab);

    //[>    cout << grey << endl;<]
    //[>    imshow("Channels", channel);<]
    //// imshow("grey", grey);
    //[>waitKey();<]

    // for (auto i = 0; i < channel.rows; i++) {
    // for (auto j = 0; j < channel.cols; j++) {
    // Vec3f cvP = channel.at<Vec3f>(i, j);
    // int intensity = grey.at<uchar>(i, j);
    /*            Vector3f p(std::min(cvP(0), 1.0f), std::min(cvP(1),
     * 1.0f),*/
    //[>std::min(cvP(2), 1.0f));<]

    // Vector3f p(cvP(0), cvP(1), cvP(2));
    // if (intensity > 0 && intensity < 85) {
    // shadows.row(count(0)) = p;
    // count(0)++;
    //}
    // else if (intensity >= 85 && intensity < 170) {
    // midtones.row(count(1)) = p;
    // count(1)++;
    //}
    // else if (intensity > 170) {
    // highlights.row(count(2)) = p;
    // count(2)++;
    //}
    //}
    //}

    /*    Matrix3f mean = Matrix3f::Zero();*/
    // Matrix3f stddev = Matrix3f::Zero();

    // vector<MatrixXf> tonesVector = {shadows, midtones, highlights};
    // if (count.sum() > 0) {
    // for (auto i = 0; i < 3; i++) {
    // auto stats = computeMeanStddev(tonesVector[i], count(i));
    // mean.row(i) = stats.first;
    // stddev.row(i) = stats.second;
    //}
    /*}*/
}

namespace EditEmbedding {

bool linearIndependent(Vector3f lhs, Vector3f rhs)
{
    float dp = lhs.dot(rhs);
    return (fabs(dp - (lhs.norm() * rhs.norm())) > 1E-5);
}

double deltaE2000(Vector3f lch1, Vector3f lch2)
{
    for (int i = 0; i < 3; i++) {
        if (fabs(lch1(i)) < 0.005) lch1(i) = 0.0;
        if (fabs(lch2(i)) < 0.005) lch2(i) = 0.0;
    }
    /*    cout << endl;*/
    // cout << lch1.transpose() << ende;
    /*cout << lch2.transpose() << endl;*/
    const static double MY_PI = 3.14159265358979323846;
    double avg_L = (lch1(0) + lch2(0)) * 0.5;
    double delta_L = lch2(0) - lch1(0);
    double avg_C = (lch1(1) + lch2(1)) * 0.5;
    double delta_C = lch1(1) - lch2(1);
    double avg_H = (lch1(2) + lch2(2)) * 0.5;
    // cout << avg_H << endl;
    assert(!isnan(avg_H));
    /*    cout << CV_PI << endl;*/
    // cout << lch1.transpose() << endl;
    /*cout << lch2.transpose() << endl;*/
    if (fabs(lch1(2) - lch2(2)) > MY_PI) avg_H += MY_PI;
    /*    cout << lch1.transpose() << endl;*/
    /*cout << lch2.transpose() << endl;*/
    double delta_H = lch2(2) - lch1(2);
    // assert(!isnan(delta_H));
    if (fabs(delta_H) > MY_PI) {
        if (lch2(2) <= lch1(2))
            delta_H += MY_PI * 2.0;
        else
            delta_H -= MY_PI * 2.0;
    }

    // assert(!isnan(delta_H));

    delta_H = sqrt(lch1(1) * lch2(1)) * sin(delta_H) * 2.0;
    double T = 1.0 - 0.17 * cos(avg_H - MY_PI / 6.0) + 0.24 * cos(avg_H * 2.0) +
               0.32 * cos(avg_H * 3.0 + MY_PI / 30.0) -
               0.20 * cos(avg_H * 4.0 - MY_PI * 7.0 / 20.0);
    double SL = avg_L - 50.0;
    SL *= SL;
    SL = SL * 0.015 / sqrt(SL + 20.0) + 1.0;
    double SC = avg_C * 0.045 + 1.0;
    double SH = avg_C * T * 0.015 + 1.0;
    double delta_Theta = avg_H / 25.0 - MY_PI * 11.0 / 180.0;
    delta_Theta = exp(delta_Theta * -delta_Theta) * (MY_PI / 6.0);
    double RT = abs(pow(avg_C, 7.0));  // JWH: Added abs as a hack
    double dividor = (RT + 6103515625.0);
    assert(!isnan(dividor));
    RT = sqrt(RT / dividor) * sin(delta_Theta) * -2.0;  // 6103515625 = 25^7

    assert(!isnan(SL));
    assert(!isnan(SC));
    assert(!isnan(SH));
    delta_L /= SL;
    delta_C /= SC;
    delta_H /= SH;

    cout << lch1.transpose() << endl;
    cout << lch2.transpose() << endl;

    assert(!isnan(delta_L));
    assert(!isnan(delta_H));
    assert(!isnan(delta_C));
    assert(!isnan(RT));
    double sum = delta_L * delta_L + delta_C * delta_C + delta_H * delta_H +
                 RT * delta_C * delta_H;
    double dist = sqrt(sum);

    assert(!isnan(dist));
    return dist;
}

using namespace tapkee;

struct MyDistanceCallback {
    ScalarType distance(Vector3f l, Vector3f r)
    {
        auto val = (l - r).squaredNorm();
        // cout << "Distance " << val << endl;
        return val;
    }
    // ScalarType distance(IndexType l, IndexType r) { return abs(l -
    // r); }
};

struct MatchKernelCallback {
    ScalarType kernel(Vector3f l, Vector3f r)
    {
        //  return inner_product(l.begin(), l.end(), r.begin(), 0,
        //  plus<int>(),
        //                       equal_to<string::value_type>());
        auto val = (l - r).squaredNorm();
        // cout << "Distance " << val << endl;
        return val;  // deltaE2000(l, r);
    }

    ScalarType distance(Vector3f l, Vector3f r)
    {
        auto val = (l - r).squaredNorm();
        // cout << "Distance " << val << endl;
        return val;
    }
};

void showEmbedding(Eigen::MatrixXd embedding, vector<Vector3f> colours,
                   vector<String> names)
{
    static int embedding_count = 0;

    float xMin = embedding.col(0).minCoeff();
    float xMax = embedding.col(0).maxCoeff();

    float yMin = embedding.col(1).minCoeff();
    float yMax = embedding.col(1).maxCoeff();

    embedding.col(0) = embedding.col(0).array() - xMin / (xMax - xMin);
    embedding.col(1) = embedding.col(1).array() - yMin / (yMax - yMin);

    int image_size = 600;
    Mat embedding2D(image_size, image_size, CV_8UC3, Scalar(255, 255, 255));
    int x = embedding(0, 0) * image_size;
    int y = embedding(0, 1) * image_size;

    string text = "Input";
    int fontFace = FONT_HERSHEY_COMPLEX_SMALL;
    double fontScale = 1;
    int thickness = 1;
    cv::Point textOrg(x + 40, y);
    cv::putText(embedding2D, text, textOrg, fontFace, fontScale,
                Scalar(0, 0, 255), thickness, 8);

    Vec3f colour = JWH::eigenLab2RGBCV(colours[0]);

    circle(embedding2D, Point(x, y), 6, Scalar(colour[2], colour[1], colour[0]),
           -1);

    x = embedding(1, 0) * image_size;
    y = embedding(1, 1) * image_size;

    colour = JWH::eigenLab2RGBCV(colours[1]);
    circle(embedding2D, Point(x, y), 6, Scalar(colour[2], colour[1], colour[0]),
           -1);
    textOrg = cv::Point(x + 40, y);
    text = "Edited";
    cv::putText(embedding2D, text, textOrg, fontFace, fontScale,
                Scalar(0, 255, 0), thickness, 8);

    for (auto i = 2; i < embedding.rows(); i++) {
        int x = embedding(i, 0) * image_size;
        int y = embedding(i, 1) * image_size;

        colour = JWH::eigenLab2RGBCV(colours[i]);
        circle(embedding2D, Point(x, y), 3,
               Scalar(colour[2], colour[1], colour[0]), -1);

        textOrg = cv::Point(x + 10, y);

        cv::putText(embedding2D, names[i - 2], textOrg, fontFace, 0.5,
                    Scalar(0, 0, 0), thickness, 8);
    }

    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%d-%m-%Y_%H-%M-%S");
    auto str = oss.str();
    // imshow(str, embedding2D);
    imwrite(str + ".png", embedding2D);

    embedding_count++;
}

Mat computeEmbedding(pair<Eigen::Vector3f, Eigen::Vector3f> edit_means,
                     vector<ChannelStat>& channelStats)
{

    /*    vector<Vector3f> indices(channelStats.size() * 10);*/

    //// tapkee::DenseMatrix distances(N, N);
    // for (int j = 0; j < 10; j++) {
    // for (int i = 0; i < channelStats.size(); i++) {
    // indices[i] = (Vector3f(channelStats[i].mean()) * (j / 255.0));
    // cout << "indices[i]  " << indices[i] << endl;
    // if (i == channelStats.size() && j < 100) i = 0;
    //}
    /*}*/

    const int N = (channelStats.size() * 3) + 1;
    vector<Vector3f> indices;
    vector<String> names;
    indices.push_back(edit_means.first);
    indices.push_back(edit_means.second);
    cout << "0 " << indices[0].transpose() << endl;
    cout << "1 " << indices[1].transpose() << endl;
    int index = 2;
    for (int i = 0; i < channelStats.size(); i++) {
        vector<Vector3f> means = channelStats[i].means();
        String filename = channelStats[i].filename();
        for (int j = 0; j < means.size(); j++) {
            Vector3f mean = means[j];
            if (mean(0) > 0) {
                cout << filename << " " << index << " " << mean.transpose()
                     << endl;
                indices.push_back(mean);
                names.push_back(filename);
                // cout << index << " " << indices[index].transpose() <<
                // " ";
                index++;
                /*                float d = deltaE2000(indices[index],
                 * edit_mean);*/
                // cout << d << endl;
                // auto p = make_pair(d, index);
                /*nearestNeighbours.push_back(p);*/
            }
        }
    }

    MatchKernelCallback distance;

    TapkeeOutput output =
        initialize()
            .withParameters(
                (method = KernelLocallyLinearEmbedding, num_neighbors = 5))
            .withKernel(distance)
            .embedUsing(indices);

    cout << "Output: " << output.embedding.cols() << " "
         << output.embedding.rows() << " -> " << output.embedding.transpose()
         << endl;

    vector<pair<float, int>> nearestNeighboursInput;
    vector<pair<float, int>> nearestNeighboursOutput;
    vector<pair<float, int>> nearestNeighboursRelative;

    for (auto i = 2; i < output.embedding.rows(); i++) {
        float d1 =
            (output.embedding.row(i) - output.embedding.row(0)).squaredNorm();
        auto p1 = make_pair(d1, i);
        nearestNeighboursInput.push_back(p1);

        float d2 =
            (output.embedding.row(i) - output.embedding.row(1)).squaredNorm();
        auto p2 = make_pair(d2, i);
        nearestNeighboursOutput.push_back(p2);

        float d3 = abs(d1 - d2);
        auto p3 = make_pair(d3, i);
        nearestNeighboursRelative.push_back(p3);
    }

    sort(nearestNeighboursInput.begin(), nearestNeighboursInput.end(),
         [](auto& left, auto& right) { return left.first < right.first; });

    sort(nearestNeighboursOutput.begin(), nearestNeighboursOutput.end(),
         [](auto& left, auto& right) { return left.first < right.first; });

    sort(nearestNeighboursRelative.begin(), nearestNeighboursRelative.end(),
         [](auto& left, auto& right) { return left.first > right.first; });

    for (auto i = 0; i < nearestNeighboursInput.size(); i++) {
        cout << "input " << nearestNeighboursInput[i].first << " "
             << nearestNeighboursInput[i].second << "  ";

        cout << "output " << nearestNeighboursOutput[i].first << " "
             << nearestNeighboursOutput[i].second << " ";

        cout << "relative " << nearestNeighboursRelative[i].first << " "
             << nearestNeighboursRelative[i].second << endl;
    }

    showEmbedding(output.embedding, indices, names);

    /*cout << output.embedding << endl;*/

    /*    for (int i = channelStats.size(); i < N; i++)*/
    /*indices[i] = Vector3f(i, i, i);*/

    /*    sort(nearestNeighbours.begin(), nearestNeighbours.end(),*/
    //[](auto& left, auto& right) { return left.first < right.first; });

    // for (auto nn : nearestNeighbours) {
    // cout << "nn " << nn.first << " " << nn.second << endl;
    //}

    // bool converged = false;
    // int num_neighbors = 1;
    // while (!converged) {

    // float error = 1.0;
    // if (num_neighbors == 1) {
    // int nn_index = nearestNeighbours[0].second;
    // Vector3f diff = (edit_mean -
    // indices[nearestNeighbours[0].second]);
    // error = diff.norm();
    //}
    // else if (num_neighbors == 2 &&
    //! linearIndependent(indices[nearestNeighbours[0].second],
    // indices[nearestNeighbours[1].second])) {

    // Vector3f low, high;
    // if (indices[nearestNeighbours[0].second].norm() <
    // indices[nearestNeighbours[1].second].norm()) {
    // low = indices[nearestNeighbours[0].second];
    // high = indices[nearestNeighbours[1].second];
    //}
    // else {
    // low = indices[nearestNeighbours[1].second];
    // high = indices[nearestNeighbours[2].second];
    //}

    // float norm = (low - high).norm();
    // cout << "Linearly dependnet todo " << endl;
    //}
    // else {
    // MatrixXf A(3, num_neighbors);
    // for (auto i = 0; i < num_neighbors; i++) {
    // int index = nearestNeighbours[i].second;
    //// cout << index << endl;
    // A.col(i) = indices[index];
    //}

    // Vector3f b = edit_mean;
    //[>        Vector3f b(0.5, 0.5, 0.5);<]
    //// A.col(0) = Vector3f(0.3, 0.3, 0.3) - b;
    //// A.col(1) = Vector3f(0.7, 0.7, 0.7) - b;
    //[>A.col(2) = Vector3f(0.9, 0.9, 0.9) - b;<]

    // auto C = A.adjoint() * A;
    // VectorXf b2(num_neighbors);
    // for (auto i = 0; i < num_neighbors; i++) {
    // b2(i) = 1;
    //}
    //// cout << C.jacobiSvd(ComputeThinU | ComputeThinV).solve(b2) <<
    //// endl;

    // MatrixXf x = C.jacobiSvd(ComputeThinU | ComputeThinV).solve(b2);

    // auto x2 = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);
    //// cout << x << endl;f

    ////// cout << A * Vector3f(0.5, 0.5, 0) << endl;

    // cout << "Here is the matrix A:\n" << A << endl;
    // cout << "Here is the matrix C:\n" << C << endl;
    // cout << "Here is the right hand side b:\n" << b << endl;
    // cout << "The least-squares solution is:\n" << x2 << endl;

    // cout << "normalized x " << x / x.sum() << endl;

    //// Vector3f recon = A * x;
    //// cout << "The recon A * x is:\n" << recon << endl;
    //// double relative_error =
    ////(recon - b).norm() / b.norm();  // norm() is L2 norm
    //[>cout << "The relative error is:\n" << relative_error << endl;<]

    // MatrixXf recon = A * x;
    // cout << "A * x2; " << recon << endl;
    // error = (recon - b).norm() / b.norm();
    //}

    // cout << num_neighbors << " error " << error << endl;
    // num_neighbors++;
    // if (error < 0.1 || num_neighbors == 5) {
    // converged = true;
    //}
    //}

    // MatchKernelCallback distance;

    /*    TapkeeOutput output =*/
    // initialize()
    //.withParameters(
    //(method = KernelLocallyLinearEmbedding, num_neighbors = 15))
    //.withKernel(distance)
    //.embedUsing(indices);

    // cout << "Output: " << output.embedding.cols() << " "
    //<< output.embedding.rows() << "  " << output.embedding.transpose()
    /*<< endl;*/

    Mat embedding;

    return embedding;
}

void addNeighbours(queue<Eigen::Vector2i>& boundryQueue, Mat& visited,
                   Eigen::Vector2i pixel)
{

    // cout << pixel.transpose() << endl;
    if (pixel(0) > 1 && visited.at<uchar>(pixel(0) - 1, pixel(1)) == 0) {
        boundryQueue.push(Eigen::Vector2i(pixel(0) - 1, pixel(1)));
        visited.at<uchar>(pixel(0) - 1, pixel(1)) = 127;
    }
    if (pixel(1) > 1 && visited.at<uchar>(pixel(0), pixel(1) - 1) == 0) {
        boundryQueue.push(Eigen::Vector2i(pixel(0), pixel(1) - 1));
        visited.at<uchar>(pixel(0), pixel(1) - 1) = 127;
    }

    if (pixel(0) < (visited.rows - 1) &&
        visited.at<uchar>(pixel(0) + 1, pixel(1)) == 0) {
        boundryQueue.push(Eigen::Vector2i(pixel(0) + 1, pixel(1)));
        visited.at<uchar>(pixel(0) + 1, pixel(1)) = 127;
    }
    if (pixel(1) < (visited.cols - 1) &&
        visited.at<uchar>(pixel(0), pixel(1) + 1) == 0) {
        boundryQueue.push(Eigen::Vector2i(pixel(0), pixel(1) + 1));
        visited.at<uchar>(pixel(0), pixel(1) + 1) = 127;
    }
}

PatchStat computeNeightbourStatistics(Mat channel, Mat mask, Mat material)
{

    Mat mask_inverted;
    cvtColor(mask, mask_inverted, COLOR_BGR2GRAY);
    mask_inverted.convertTo(mask_inverted, CV_8UC1, 255.0);

    Mat channelLab;
    cvtColor(channel, channelLab, CV_BGR2Lab);

    /*    imshow("visited1", visited);*/
    /*    imshow("beautyPass", beautyPass);*/
    /*waitKey();*/

    int current_pixel_count = 0;
    int max_pixels = channel.rows * channel.cols;
    Eigen::MatrixXf allPixels(max_pixels, 3);

    Mat visited(channel.rows, channel.cols, CV_8UC1, Scalar(0));

    for (int i = 0; i < channel.rows; i++) {
        for (int j = 0; j < channel.cols; j++) {
            if (material.at<uchar>(i, j) == 255 &&
                mask_inverted.at<uchar>(i, j) == 0) {
                Vec3f val = channelLab.at<Vec3f>(i, j);
                if (val[0] > 2) {
                    allPixels.row(current_pixel_count) =
                        Eigen::Vector3f(val[0], val[1], val[2]);
                    visited.at<uchar>(i, j) = 255;
                    current_pixel_count++;
                }
            }
        }
    }
    PatchStat neighbourStats;

    if (current_pixel_count == 0) {
        cout << "No pixels found to compute edit mean " << endl;
        return neighbourStats;
    }

    /*    imshow("visited", visited);*/
    /*waitKey();*/

    // cout << allPixels.block(0, 0, current_pixel_count, 3) << endl;

    pair<Vector3f, Vector3f> inputMeanStddev =
        JWH::computeMeanStddev(allPixels, current_pixel_count);

    neighbourStats.mean = inputMeanStddev.first;
    neighbourStats.stddev = inputMeanStddev.second;

    return neighbourStats;
}

/// Region Gowing
//    /*    imshow("visited1", visited);*/
/*    imshow("beautyPass", beautyPass);*/
/*waitKey();*/

/*    int masked_pixel_count = 0;*/
// queue<Eigen::Vector2i> boundryQueue;
// for (int i = 0; i < visited.rows; i++) {
// for (int j = 0; j < visited.cols; j++) {
// if (visited.at<uchar>(i, j) == 255) {
// addNeighbours(boundryQueue, visited, Eigen::Vector2i(i, j));
// masked_pixel_count++;
//}
//}
//}

// int target_region = masked_pixel_count * 8;
// int current_pixel_count = 0;
// Eigen::MatrixXf allPixels(target_region, 3);

// while (boundryQueue.size() > 0 && target_region > current_pixel_count) {
// Eigen::Vector2i pixel = boundryQueue.front();
// boundryQueue.pop();

// Vec3f val = channelLab.at<Vec3f>(pixel(0), pixel(1));
// if (val[0] > 10) {
// allPixels.row(current_pixel_count) =
// Eigen::Vector3f(val[0], val[1], val[2]);

// visited.at<uchar>(pixel(0), pixel(1)) = 200;
//// cout << allPixels.row(current_pixel_count) << endl;
// current_pixel_count++;
//}
// addNeighbours(boundryQueue, visited, pixel);
//}

//[>    cout << "target_region " << target_region << endl;<]
//// cout << "current_pixel_count " << current_pixel_count << endl;

/*imshow("visited1", visited);*/
}
