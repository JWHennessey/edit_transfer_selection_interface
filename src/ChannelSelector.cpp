#include "ChannelSelector.h"
//#include <omp.h>
#include <Eigen/Eigenvalues>
#include <thread>
#include "MaskEdit.h"
#include "Segmentor.h"

ChannelSelector::ChannelSelector() {}

void ChannelSelector::create(shared_ptr<vector<String>> filenames,
                             shared_ptr<vector<Mat>> imageMats,
                             shared_ptr<vector<Mat>> materialIdMats)

{

    mFilenames = filenames;
    mImageMats = imageMats;
    mMaterialIdMats = materialIdMats;
    // First filename should allways be rbb e.g. _.exr
    String str = (*mFilenames)[0];
    assert(str.find("/_.exr") != string::npos);
    mFolder = str.substr(0, str.size() - 5);
    init();
}

bool sort_pair(const std::pair<double, int> &lhs,
               const std::pair<double, int> &rhs)
{
    return lhs.first > rhs.first;
}

PatchStat computeNoneBlackMeanStddev(Mat image)
{
    int total = image.cols * image.rows;
    MatrixXf allPixels(total, 3);
    int count = 0;
    for (auto x = 0; x < image.rows; x++) {
        for (auto y = 0; y < image.cols; y++) {
            Vec3f p = image.at<Vec3f>(x, y);
            Vector3f pixel(p[0], p[1], p[2]);
            if (pixel.norm() > 0.1) {
                allPixels.row(count) = pixel;
                count++;
            }
        }
    }
    auto p = computeMeanStddev(allPixels, count);
    PatchStat selectedStat;
    selectedStat.mean = p.first;
    selectedStat.stddev = p.second;
    return selectedStat;
}

float patchDistance(PatchStat p1, PatchStat p2)
{
    Vector3f m = p1.mean - p2.mean;
    Vector3f l = p1.mean.array().log() - p2.mean.array().log();
    Vector3f s = p1.stddev - p2.stddev;
    Vector3f g = p1.gradient_mean - p2.gradient_mean;
    float dist = m.norm() + s.norm() + g.norm();

    /*    if (!std::isnan(l.norm()) && !std::isinf(l.norm())) {*/
    // dist += l.norm();
    /*}*/

    return dist;
}

void ChannelSelector::findNeighbouringPatches(vector<Mat> &neighbouringPatches,
                                              Mat polygon)
{

    Mat polygonBinary;
    cvtColor(polygon, polygonBinary, CV_BGR2GRAY);
    Mat Points;
    findNonZero(polygonBinary, Points);
    Rect rect = boundingRect(Points);

    Vector2i position(rect.x + rect.width * 0.5, rect.y + rect.height * 0.5);
    int radius = (Vector2i(rect.x, rect.y) -
                  Vector2i(rect.x + rect.width, rect.y + rect.height))
                     .norm();

    Mat radiusImage(polygonBinary.rows, polygonBinary.cols, CV_8UC3,
                    Scalar(0, 0, 0));

    circle(radiusImage, Point(position(0), position(1)), radius,
           Scalar(255, 255, 255), -1);

    std::cout << "Radius may be too big  " << std::endl;

    double min, max;
    for (auto candidate : mCandidatePatches) {
        /// std::cout << polygon.type() << " " << candidate.type() << std::endl;
        Mat output;
        bitwise_and(candidate, radiusImage, output);
        output -= polygon;
        minMaxLoc(output, &min, &max);
        if (max > 0) {
            cvtColor(output, output, CV_BGR2GRAY);
            neighbouringPatches.push_back(output);
        }
    }
}

void ChannelSelector::selectUniqueChannel(Mat *polygon,
                                          vector<pair<float, int>> &ordering,
                                          float &total_energy)
{
    cout << "ChannelSelector::selectChannel" << endl;

    PatchStat blackPatch;
    std::list<std::pair<double, int>> score_list;

    vector<Mat> neighbouringPatches;
    findNeighbouringPatches(neighbouringPatches, *polygon);

    VectorXd distanceVector = VectorXd::Zero(mImageMats->size());
    Scalar mean, stddev;
    Scalar meanDepth, stddevDepth;
    Scalar meanCandidateDepth, stddevCandidateDepth;
    Mat poly_mask;
    cvtColor(*polygon, poly_mask, CV_BGR2GRAY);
    for (auto i = 0; i < mGradientImages.size(); i++) {

        // For the selection region compute patch stats
        PatchStat selectedStat;
        meanStdDev((*mImageMats)[i + 1], mean, stddev, poly_mask);
        selectedStat.mean = Vector3f(mean(0), mean(1), mean(2));
        selectedStat.stddev = Vector3f(stddev(0), stddev(1), stddev(2));
        meanStdDev(mGradientImages[i], mean, stddev, poly_mask);
        selectedStat.gradient_mean = Vector3f(mean(0), mean(1), mean(2));

        //meanStdDev(mDepthImage, meanDepth, stddevDepth, poly_mask);

        cout << (*mFilenames)[i + 1] << endl;
        cout << "selectedStat.mean " << selectedStat.mean.norm() << " "
             << selectedStat.stddev.norm() << endl;

        auto found = (*mFilenames)[i + 1].find("_.Lighting.exr");

        if (selectedStat.mean.norm() < 0.001 || found != string::npos) {
            distanceVector(i) = 0;
            continue;
        }
        for (auto candidate : neighbouringPatches) {
            // Mat candidate_region;
            // copyTo(candidate_region, candidate);
            PatchStat candidateStat;
            meanStdDev((*mImageMats)[i + 1], mean, stddev, candidate);
            candidateStat.mean = Vector3f(mean(0), mean(1), mean(2));
            candidateStat.stddev = Vector3f(stddev(0), stddev(1), stddev(2));
            meanStdDev(mGradientImages[i], mean, stddev, candidate);
            candidateStat.gradient_mean = Vector3f(mean(0), mean(1), mean(2));

            /*            meanStdDev(mDepthImage, meanCandidateDepth,
             * stddevCandidateDepth,*/
            // poly_mask);

            // int diff = (meanCandidateDepth[0] - meanDepth[0]);
            /*if (diff > 20) continue;*/

            distanceVector(i) += patchDistance(selectedStat, candidateStat);
        }
    }

    std::cout << distanceVector << std::endl;

    MatrixXd C = distanceVector * distanceVector.transpose();

    SelfAdjointEigenSolver<MatrixXd> eigensolver(C);

    /*    std::cout << eigensolver.eigenvalues() << std::endl;*/
    /*std::cout << eigensolver.eigenvectors() << std::endl;*/

    int max_col_index = 1;
    float max_val = eigensolver.eigenvalues()(0);
    for (int i = 1; i < mImageMats->size(); i++) {
        if (eigensolver.eigenvalues()(i) > max_val) {
            max_val = eigensolver.eigenvalues()(i);
            max_col_index = i;
        }
    }

    total_energy = 0;
    for (int i = 0; i < mGradientImages.size(); i++) {
        score_list.push_back(
            make_pair(eigensolver.eigenvectors().col(max_col_index)(i), i));
        total_energy += (float)eigensolver.eigenvectors().col(max_col_index)(i);
    }

    score_list.sort(sort_pair);
    ordering.clear();

    for (auto p : score_list) {
        cout << (*mFilenames)[p.second + 1] << " " << p.first << endl;
        ordering.push_back(p);
    }
}

// Rect r = convertToCVRect(rect, (*mImageMats)[0]);
/*    int nloop = mChannelPatchMasks.size();*/
// size_t nthreads = std::thread::hardware_concurrency();
//{
//// Pre loop
// std::cout << "parallel (" << nthreads << " threads):" << std::endl;
// std::vector<std::thread> threads(nthreads);
// std::mutex critical;
// nthreads = size_t(1);
// for (int t = 0; t < nthreads; t++) {
// threads[t] = std::thread(std::bind(
//[&](const int bi, const int ei, const int t) {
//// mChannelPatchMasks.size()
// for (auto i = bi; i < ei; i++) {

//// For the selection region compute patch stats
// Mat selected_region = (*mImageMats)[i + 1](r);
// PatchStat selectedStat =
// computeNoneBlackMeanStddev(selected_region);

// Mat selected_region_gradient = mGradientImages[i](r);

// PatchStat selectedStatGradient =
// computeNoneBlackMeanStddev(
// selected_region_gradient);

// selectedStat.gradient_mean = selectedStatGradient.mean;

// vector<PatchStat> neighbouringPatches;
// getNeighbouringPatchStats(neighbouringPatches,
// mChannelPatchMasks[i],
// mChannelPatchStats[i], r, i);

// neighbouringPatches.push_back(blackPatch);

// int total_patches = neighbouringPatches.size();
// Mat distances(total_patches, total_patches, CV_32FC1,
// Scalar(0));

// VectorXd distanceVector(total_patches);
// for (auto a = 0; a < neighbouringPatches.size(); a++) {
// PatchStat neighbourStatA = neighbouringPatches[a];
// float d =
// patchDistance(neighbourStatA, selectedStat);
// distanceVector(a) = d;
//}

// MatrixXd C =
// distanceVector * distanceVector.transpose();

// SelfAdjointEigenSolver<MatrixXd> eigensolver(C);

// score_list.push_back(
// make_pair(eigensolver.eigenvalues().maxCoeff(), i));
//}

//},
// t * nloop / nthreads,
//(t + 1) == nthreads ? nloop : (t + 1) * nloop / nthreads, t));
//}
// std::for_each(threads.begin(), threads.end(),
//[](std::thread &x) { x.join(); });
/*}*/

void ChannelSelector::getNeighbouringPatchStats(
    vector<PatchStat> &neighbouringPatches, const vector<Mat> &patchMasks,
    const vector<PatchStat> &patchStats, Mat *polygon, int channelId)
{

    Mat polygonBinary;
    cvtColor(*polygon, polygonBinary, CV_BGR2GRAY);
    Mat Points;
    findNonZero(polygonBinary, Points);
    Rect rect = boundingRect(Points);

    Vector2i position(rect.x + rect.width * 0.5, rect.y + rect.height * 0.5);
    int radius = (Vector2i(rect.x, rect.y) -
                  Vector2i(rect.x + rect.width, rect.y + rect.height))
                     .norm();

    for (auto i = 0; i < patchStats.size(); i++) {
        PatchStat ps = patchStats[i];
        int dist = (ps.centroid - position).norm();
        if (dist <= radius) {
            Mat region = patchMasks[i];
            Mat partialRegion;
            region.copyTo(partialRegion, *polygon);
            Scalar m = mean(partialRegion);
            // If the seclection and the neigbouring patch intersect, then
            // substract that region from the nighbour
            if (m[0] > 0) {
                Mat channelWithRegionRemoved = region - partialRegion;

                (*mImageMats)[channelId + 1].copyTo(channelWithRegionRemoved,
                                                    region);
                m = mean(channelWithRegionRemoved);
                if (m[0] > 0) {
                    ps = computeNoneBlackMeanStddev(channelWithRegionRemoved);
                }
                else {
                    continue;
                }
            }

            Mat selected_region_gradient = mGradientImages[channelId];
            selected_region_gradient.copyTo(selected_region_gradient, *polygon);

            PatchStat selectedStatGradient =
                computeNoneBlackMeanStddev(selected_region_gradient);

            ps.gradient_mean = selectedStatGradient.mean;

            neighbouringPatches.push_back(ps);
        }
    }
}

void ChannelSelector::selectRelativeChangePatches(Mat mask, Mat beforeEdit,
                                                  Mat afterEdit,
                                                  vector<Mat> &labels)
{
    /*    std::list<std::pair<double, int>> distance_change_list;*/

    // Mat points;
    // Mat maskSingleChannel;
    // cvtColor(mask, maskSingleChannel, CV_BGR2GRAY);
    // maskSingleChannel.convertTo(maskSingleChannel, CV_8UC1, 255);
    // threshold(maskSingleChannel, maskSingleChannel, 20, 255, THRESH_BINARY);
    //[>    imshow("mask", maskSingleChannel);<]
    //// cout << "maskSingleChannel.type() " << maskSingleChannel.type() <<
    /// endl;

    //[>waitKey();<]
    // findNonZero(maskSingleChannel, points);

    // Rect rect = boundingRect(points);

    // Vector2i position(rect.x + rect.width * 0.5, rect.y + rect.height * 0.5);
    // int radius = (Vector2i(rect.x, rect.y) -
    // Vector2i(rect.x + rect.width, rect.y + rect.height))
    //.norm();

    // cout << "Radius " << radius << endl;

    // cout << "mChannelPatchStats.size() " << mChannelPatchStats.size() <<
    // endl;

    // vector<Mat> neighbouringPatches = {maskSingleChannel};
    // for (auto i = 0; i < mChannelPatchStats.size(); i++) {
    // for (auto j = 0; j < mChannelPatchStats[i].size(); j++) {
    // PatchStat ps = mChannelPatchStats[i][j];
    // int dist = (ps.centroid - position).norm();
    // if (dist <= radius) {
    // neighbouringPatches.push_back(mChannelPatchMasks[i][j]);
    //}
    //}
    //}

    // cout << "neighbouringPatches.size() " << neighbouringPatches.size() <<
    // endl;

    // computeDistanceChanges(neighbouringPatches, beforeEdit, afterEdit,
    // distance_change_list);

    // distance_change_list.sort(sort_pair);

    // std::vector<std::pair<double, int>> orderedDistances{
    // std::begin(distance_change_list), std::end(distance_change_list)};

    // const static int NO_NEIGHBOURS = 1;
    // for (auto i = 0; i < NO_NEIGHBOURS; i++) {
    // Mat mask = neighbouringPatches[orderedDistances[i].second];
    // labels.push_back(mask);
    /*}*/
}

// Private
void ChannelSelector::init()
{
    std::vector<String> candidatePatchesFiles;
    glob(mFolder + "candidate_patches/", candidatePatchesFiles);
    std::cout << "candidatePatchesFiles.size() " << candidatePatchesFiles.size()
              << std::endl;

    for (auto i = 1; i < mImageMats->size(); i++) {

        std::cout << "Image " << i << std::endl;
        Mat m = (*mImageMats)[i];

        // create intensity_images
        Mat greyscaled;
        m.convertTo(greyscaled, CV_8UC3, 255);
        cv::cvtColor(greyscaled, greyscaled, CV_BGR2GRAY);
        mIntensityImages.push_back(greyscaled);

        // create gradient_images
        Mat gradients;
        Laplacian(greyscaled, gradients, CV_16S, 5, 1.0);
        gradients.convertTo(gradients, CV_8UC1);
        cv::cvtColor(gradients, gradients, CV_GRAY2BGR);
        gradients.convertTo(gradients, CV_32FC3, 1.0 / 255.0);
        mGradientImages.push_back(gradients);

        if (candidatePatchesFiles.size() <= 1) {

            for (auto j = 0; j < mMaterialIdMats->size(); j++) {

                std::cout << "mMaterialIdMats " << j << std::endl;
                Mat channel_for_material;
                greyscaled.copyTo(channel_for_material, (*mMaterialIdMats)[j]);

                Mat binary_channel;
                threshold(channel_for_material, binary_channel, 1, 255,
                          THRESH_BINARY);

                Mat stats;
                Mat centroids;
                Mat connected_components;
                int no_cc = connectedComponentsWithStats(
                    binary_channel, connected_components, stats, centroids);

                cout << "no_cc " << no_cc << endl;
                for (auto k = 1; k < no_cc; k++) {
                    Mat patch(greyscaled.rows, greyscaled.cols, CV_8UC1,
                              Scalar(0));
                    int patch_pixel_count = 0;
                    int total_pixels = greyscaled.rows * greyscaled.cols;
                    for (auto r = 0; r < greyscaled.rows; r++) {
                        for (auto c = 0; c < greyscaled.cols; c++) {
                            int val = connected_components.at<int>(r, c);
                            if (val == k) {
                                patch.at<uchar>(r, c) = 255;
                                patch_pixel_count++;
                            }
                        }
                    }

                    if (patch_pixel_count > 200) {
                        mCandidatePatches.push_back(patch);
                        cv::imwrite(mFolder + "candidate_patches/Patch_" +
                                        to_string(mCandidatePatches.size()) +
                                        ".png",
                                    patch);
                    }
                }
            }
        }
    }

    if (candidatePatchesFiles.size() > 1) {
        for (auto file : candidatePatchesFiles) {
            Mat candidate = imread(file, CV_LOAD_IMAGE_COLOR);
            if (candidate.cols > 0 && candidate.rows > 0) {
                mCandidatePatches.push_back(candidate);
            }
        }
    }
    std::cout << "mCandidatePatches.size() " << mCandidatePatches.size()
              << std::endl;
    // read depth_images
    //mDepthImage = imread(mFolder + "/depth/_.Depth.exr", cv::IMREAD_UNCHANGED);
    //mDepthImage.convertTo(mDepthImage, CV_8UC3, 255.0);
    /*cv::cvtColor(mDepthImage, mDepthImage, CV_BGR2GRAY);*/
}

void ChannelSelector::getNeighbouringPatches(JWH::Rectangle &rect,
                                             vector<cv::Rect> &sampleRectangles)
{

    Mat rgb_image = (*mImageMats)[0];

    auto current_annotation = JWH::convertToCVRect(rect, rgb_image);

    sampleRectangles.push_back(current_annotation);
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            if ((x == 0 && y == 0)) continue;
            Rect local_sample = current_annotation;
            local_sample.x += local_sample.width * float(y);   // * 0.5;
            local_sample.y += local_sample.height * float(x);  // * 0.5;

            if (local_sample.x < 0) continue;  // local_sample.x = 0;

            if (local_sample.y < 0) continue;  // local_sample.y = 0;

            if (local_sample.x > rgb_image.cols)
                continue;  // local_sample.x = rgb_image.cols;

            if (local_sample.y > rgb_image.rows)
                continue;  // local_sample.y = rgb_image.rows;

            int end_x = local_sample.x + local_sample.width;
            int end_y = local_sample.y + local_sample.height;

            if (end_x > rgb_image.cols)
                continue;  // local_sample.width = local_sample.width -
                           // (end_x -
                           // rgb_image.cols);

            if (end_y > rgb_image.rows)
                continue;  // local_sample.height = local_sample.height -
                           // (end_y
                           // -
                           // rgb_image.rows);
            sampleRectangles.push_back(local_sample);
        }
    }
}

/*void ChannelSelector::getDepthHistograms(vector<cv::Rect>
 * &sampleRectangles,*/
// vector<Mat> &depthRectangleHistograms,
// int &histogram_size)
//{
// for (auto j = 0; j < sampleRectangles.size(); j++) {
// Rect sample_annotation = sampleRectangles[j];
// Mat depth = mDepthImage(sample_annotation);
// Mat depth_hist;
// calcHist(&depth, 1, 0, Mat(), depth_hist, 1, &histSize, &histRange,
// uniform, accumulate_hist);
// depthRectangleHistograms.push_back(depth_hist);
//}

// Mat depth_cropped = mDepthImage(sampleRectangles[0]);
// Scalar depthMean, depthStddev;
// meanStdDev(depth_cropped, depthMean, depthStddev);
// if (depthStddev[0] > 45) histogram_size = 3;  // Don't include depth

// assert(sampleRectangles.size() == depthRectangleHistograms.size());
/*}*/

void matToNormalizedLaplacian(Mat &distanceMatrix)
{
    Mat column_sum;
    cv::reduce(distanceMatrix, column_sum, 0, CV_REDUCE_SUM, CV_32FC1);
    column_sum = 1. / column_sum;
    sqrt(column_sum, column_sum);

    Mat D = Mat::zeros(distanceMatrix.rows, distanceMatrix.cols, CV_32FC1);
    for (auto i = 0; i < distanceMatrix.rows; i++)
        D.at<float>(i, i) = column_sum.at<float>(i);

    distanceMatrix = D * distanceMatrix * D;
}

void ChannelSelector::computeDistanceMatrixes(
    vector<Mat> &distanceMatrices, vector<cv::Rect> &sampleRectangles,
    vector<Mat> &depthRectangleHistograms, int &histogram_size)

{
    for (size_t i = 0; i < mIntensityImages.size(); i++) {
        Mat distanceMatrix(sampleRectangles.size(), sampleRectangles.size(),
                           CV_32FC1, Scalar(0));

        {  // Remove patches with weak signal
            Mat intensity_cropped = mIntensityImages[i](sampleRectangles[0]);

            Scalar intensityMean, intensityStddev;
            meanStdDev(intensity_cropped, intensityMean, intensityStddev);

            if ((fabs(intensityMean[0] - intensityStddev[0]) < 1.0f &&
                 intensityMean[0] < 2.0f) ||
                intensityMean[0] < 0.5) {
                /*cout << mFilenames[i + 1] << " Removed " <<
                 * intensityStddev[0]*/
                /*<< " " << intensityMean[0] << endl;*/
                distanceMatrices.push_back(distanceMatrix);
                continue;
            }
        }

        for (auto j = 0; j < sampleRectangles.size(); j++) {
            for (auto k = (j + 1); k < sampleRectangles.size(); k++) {

                Rect sample_annotation_a = sampleRectangles[j];
                Rect sample_annotation_b = sampleRectangles[k];

                Mat intensity_a = mIntensityImages[i](sample_annotation_a);
                Mat intensity_b = mIntensityImages[i](sample_annotation_b);

                Mat gradient_a = mGradientImages[i](sample_annotation_a);
                Mat gradient_b = mGradientImages[i](sample_annotation_b);

                Mat intensity_a_hist;
                /// Compute the histograms:
                calcHist(&intensity_a, 1, 0, Mat(), intensity_a_hist, 1,
                         &histSize, &histRange, uniform, accumulate_hist);

                Mat intensity_b_hist;
                /// Compute the histograms:
                calcHist(&intensity_b, 1, 0, Mat(), intensity_b_hist, 1,
                         &histSize, &histRange, uniform, accumulate_hist);

                Mat gradient_a_hist;
                calcHist(&gradient_a, 1, 0, Mat(), gradient_a_hist, 1,
                         &histSize, &histRange, uniform, accumulate_hist);

                Mat gradient_b_hist;
                calcHist(&gradient_b, 1, 0, Mat(), gradient_b_hist, 1,
                         &histSize, &histRange, uniform, accumulate_hist);

                Mat depth_a_hist = depthRectangleHistograms[j];
                Mat depth_b_hist = depthRectangleHistograms[k];

                // histogram_size = 4;
                Mat sig1(256, histogram_size, CV_32FC1);
                Mat sig2(256, histogram_size, CV_32FC1);
                float depth_dist = 0.0;
                for (auto i = 0; i < 256; i++) {
                    sig1.at<float>(i, 0) = i;
                    sig2.at<float>(i, 0) = i;
                    sig1.at<float>(i, 1) = intensity_a_hist.at<float>(i);
                    sig2.at<float>(i, 1) = intensity_b_hist.at<float>(i);
                    sig1.at<float>(i, 2) = gradient_a_hist.at<float>(i);
                    sig2.at<float>(i, 2) = gradient_b_hist.at<float>(i);
                    if (histogram_size > 3) {
                        sig1.at<float>(i, 3) = depth_a_hist.at<float>(i);
                        sig2.at<float>(i, 3) = depth_b_hist.at<float>(i);
                        depth_dist = norm(depth_a_hist - depth_b_hist);
                    }
                }

                float emd_dist = EMD(sig1, sig2, CV_DIST_L2);
                // float emd_dist = compareHist( sig1, sig2,
                // CV_COMP_BHATTACHARYYA );
                // float emd_dist = 10.0 * norm(edge_a_hist - edge_b_hist) +
                // norm(intensity_a_hist - intensity_b_hist) + depth_dist;

                distanceMatrix.at<float>(j, k) = emd_dist;
                distanceMatrix.at<float>(k, j) = emd_dist;
            }
        }
        if (sampleRectangles.size() > 2) {
            matToNormalizedLaplacian(distanceMatrix);
        }
        distanceMatrices.push_back(distanceMatrix);
    }
}

void create1DRGBHistogramSig(Mat &sig, Mat rgb)
{
    float range[] = {0, 256};
    const float *histRange = {range};

    bool uniform = true;
    bool accumulate = false;

    vector<Mat> bgr_planes;
    split(rgb, bgr_planes);

    // for (auto i = 0; i < 3; i++) {
    Mat hist;
    calcHist(&rgb, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform,
             accumulate);
    for (auto j = 0; j < 256; j++) {
        sig.at<float>(j, 0) = j;
        sig.at<float>(j, 1) = hist.at<float>(j);
    }
    // }
}

void ChannelSelector::computeDistanceChanges(
    vector<Mat> &neighbouringPatches, Mat beforeEdit, Mat afterEdit,
    std::list<std::pair<double, int>> &distance_change_list)
{
    cout << "computeDistanceChanges " << neighbouringPatches.size() << endl;
    Mat afterRoiCropped;
    afterEdit.copyTo(afterRoiCropped, neighbouringPatches[0]);

    Mat beforeRoiCropped;
    beforeEdit.copyTo(beforeRoiCropped, neighbouringPatches[0]);

    /*    Mat afterRoiCroppedRGB;*/
    /*afterRoiCropped.convertTo(afterRoiCroppedRGB, CV_8UC3, 255);*/

    /*    imshow("afterRoiCropped", afterRoiCropped);*/
    /*waitKey();*/

    PatchStat afterMean = computeNoneBlackMeanStddev(afterRoiCropped);
    /*    cv::cvtColor(afterRoiCroppedRGB, afterRoiCroppedRGB,
     * CV_BGR2GRAY);*/
    // Mat afterSig(256, 2, CV_32FC1, Scalar(0));
    /*create1DRGBHistogramSig(afterSig, afterRoiCroppedRGB);*/
    // imshow("After", afterRoiCroppedRGB);

    // cout << afterSig.rows << " " << afterSig.cols << endl;

    // Mat beforeRoiCroppedRGB;
    // beforeRoiCropped.convertTo(beforeRoiCroppedRGB, CV_8UC3, 255);
    PatchStat beforeMean = computeNoneBlackMeanStddev(beforeRoiCropped);

    /*    imshow("beforeRoiCroppedRGB", beforeRoiCropped);*/
    /*waitKey();*/

    /*    cv::cvtColor(beforeRoiCroppedRGB, beforeRoiCroppedRGB,
     * CV_BGR2GRAY);*/
    // Mat beforeSig(256, 2, CV_32FC1, Scalar(0));
    /*create1DRGBHistogramSig(beforeSig, beforeRoiCroppedRGB);*/
    // cout << beforeSig.rows << " " << beforeSig.cols << endl;
    // imshow("Before", beforeRoiCroppedRGB);

    for (auto j = 1; j < neighbouringPatches.size(); j++) {
        Mat mask = neighbouringPatches[j];

        // cout << j << " " << mask.type() << endl;
        // Mat maskSingleChannel;
        // cvtColor(mask, maskSingleChannel, CV_BGR2GRAY);
        // maskSingleChannel.convertTo(maskSingleChannel, CV_8UC1, 255);
        /*threshold(maskSingleChannel, maskSingleChannel, 20, 255,
         * THRESH_BINARY);*/

        Mat roiCropped;
        beforeEdit.copyTo(roiCropped, mask);

        /*        Mat roiCroppedRGB;*/
        /*roiCropped.convertTo(roiCroppedRGB, CV_8UC3, 255);*/

        PatchStat roiMean = computeNoneBlackMeanStddev(roiCropped);

        float dist1 = patchDistance(afterMean, roiMean);
        float dist2 = patchDistance(beforeMean, roiMean);
        float diff = fabs(dist1 - dist2) / dist1;
        auto p = make_pair(diff, j);
        distance_change_list.push_back(p);
        // cout << dist2 << " " << dist1 << " " << diff << endl;
    }
}

// Util functions
double outlierScore(Mat &distanceMatrix)
{
    double min_val = 1000;
    Mat knn(distanceMatrix.cols, 3, CV_32FC1, Scalar(100));
    for (auto i = 1; i < distanceMatrix.cols; i++) {
        double val = distanceMatrix.at<float>(0, i);
        if (min_val > val) min_val = val;
    }
    return min_val;
}

void findOutlier(vector<Mat> distanceMatrices,
                 std::list<std::pair<double, int>> &score_list)
{
    for (auto i = 0; i < distanceMatrices.size(); i++) {
        if (sum(distanceMatrices[i])[0] == 0.0) {
            score_list.push_back(std::make_pair(-1000, i));
            continue;
        }
        double score = outlierScore(distanceMatrices[i]);
        score_list.push_back(std::make_pair(score, i));
    }
    score_list.sort(sort_pair);
}
