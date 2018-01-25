#include "Segmentor.h"

Segmentor::Segmentor(Mat &channel, cv::Mat polygonMask)
    : mChannel(channel), mPolygonMask(polygonMask)
{
    mMask = segmentGrabCut();
}

Mat Segmentor::getMask() { return mMask; }

Mat Segmentor::segmentGrabCut()
{
    Mat bgdModel, fgdModel, segmentation, image;
    mChannel.convertTo(image, CV_8UC3, 255);

    /*    cout << "mChannel.channels() " << mChannel.channels() << endl;*/
    // cout << "mChannel.type() " << mChannel.type() << endl;
    //// cout << image << endl;
    // Scalar mean, stddev;
    // meanStdDev(mChannel, mean, stddev);
    // cout << "mChanel " << mean << " " << stddev << endl;
    // meanStdDev(image, mean, stddev);
    // cout << "image " << mean << " " << stddev << endl;

    // imshow("mChannel", mChannel);
    /*imshow("RGB", image);*/

    Mat binaryPolygon;
    cvtColor(mPolygonMask, binaryPolygon, CV_BGR2GRAY);
    binaryPolygon.convertTo(binaryPolygon, CV_8UC1, 255);
    threshold(binaryPolygon, binaryPolygon, 1, 255, CV_THRESH_BINARY);

    Mat dt;
    distanceTransform(binaryPolygon, dt, CV_DIST_L2, 3);
    normalize(dt, dt, 0, 1, NORM_MINMAX);

    segmentation.create(image.size(), CV_8UC1);
    for (auto i = 0; i < segmentation.rows; i++) {
        for (auto j = 0; j < segmentation.rows; j++) {
            if (binaryPolygon.at<uchar>(i, j) == 0) {
                segmentation.at<uchar>(i, j) = GC_BGD;
            }
            else if (dt.at<float>(i, j) < 0.1) {
                segmentation.at<uchar>(i, j) = GC_PR_BGD;
                binaryPolygon.at<uchar>(i, j) = 50;
            }
            else if (dt.at<float>(i, j) > 0.8) {
                segmentation.at<uchar>(i, j) = GC_FGD;
            }
            else {
                segmentation.at<uchar>(i, j) = GC_PR_FGD;
                binaryPolygon.at<uchar>(i, j) = 127;
            }
        }
    }

    /* imshow("binaryPolygon", binaryPolygon);*/
    // imshow("mPolygonMask", mPolygonMask);
    // imshow("distanceTransform", dt);
    /*waitKey();*/

    grabCut(image, segmentation, Rect(), bgdModel, fgdModel, 10,
            GC_INIT_WITH_MASK);

    // Mat outSegmentation(image.size(), CV_32FC3, Scalar(0, 0, 0));
    for (auto i = 0; i < segmentation.rows; i++) {
        for (auto j = 0; j < segmentation.rows; j++) {
            if (segmentation.at<uchar>(i, j) == GC_BGD) {
                // segmentation.at<uchar>(i, j) = 0;
            }
            else if (segmentation.at<uchar>(i, j) == GC_PR_BGD) {
                // segmentation.at<uchar>(i, j) = 50;
            }
            else if (segmentation.at<uchar>(i, j) == GC_PR_FGD) {
                // segmentation.at<uchar>(i, j) = 127;
            }
            else {
                // segmentation.at<uchar>(i, j) = 255;
            }
        }
    }

    /*    imshow("segmentation", segmentation);*/
    // imshow("outSegmentation", outSegmentation);
    /*waitKey();*/
    return segmentation;
}
